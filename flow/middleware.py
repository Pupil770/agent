from langchain.agents.middleware import after_model, before_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config

from flow.state import ConversationStage, get_state

# ── 关键词集合 ──────────────────────────────────────────────

_POSITIVE = {"好", "好的", "谢谢", "感谢", "没问题", "解决了", "可以了", "明白了", "懂了", "是的", "满意"}
_NEGATIVE = {"没有", "不是", "不行", "还是不行", "没解决", "没帮助", "不明白", "不懂",
             "不合理", "差", "差评", "投诉", "不满意", "还是不对", "没弄好"}
_HUMAN_REQUEST = {"转人工", "人工客服", "人工服务", "真人", "接线员", "找人工", "叫人工", "联系人工"}
_QUESTION_HINTS = {"吗", "怎么", "如何", "什么", "为什么", "哪", "多少", "能不能", "可不可以", "帮我", "请问"}

# ── 多轮对话配置 ────────────────────────────────────────────

MAX_TURNS = 30                    # 最大对话轮次，超过自动结束
RECENT_WINDOW = 6                 # 保留最近 N 轮完整消息（1轮 = 1条Human + 1条AI）
TOKEN_SOFT_LIMIT = 6000           # token 软上限，超出时触发摘要压缩
_CONFIRM_EVERY_N_TURNS = 3        # 每隔几轮主动确认满意度

# ── 阶段指令模板 ────────────────────────────────────────────

_STAGE_INSTRUCTIONS: dict[ConversationStage, str] = {
    ConversationStage.GREETING: (
        "【流程指令 - 首次接触】\n"
        "你必须先向用户打招呼并主动询问需要什么帮助。例如：'您好！我是智能客服助手，请问有什么可以帮您？'\n"
        "不要直接回答任何问题，先完成问候再进入解答。"
    ),
    ConversationStage.INQUIRY: (
        "【流程指令 - 问题确认】\n"
        "用户的意图尚不明确，你需要：\n"
        "1. 引导用户清晰描述问题，必要时追问细节\n"
        "2. 确认你理解了用户的核心需求后再回答\n"
        "3. 如果用户问题已足够清晰，直接使用知识库搜索工具回答，然后进入回答阶段"
    ),
    ConversationStage.ANSWERING: (
        "【流程指令 - 回答问题】\n"
        "优先使用知识库搜索工具查找答案。回答时：\n"
        "1. 基于知识库内容回答，不要编造信息\n"
        "2. 如果知识库中没有相关内容，如实告知\n"
        "3. 回答完毕后关注用户是否满意"
    ),
    ConversationStage.CONFIRMING: (
        "【流程指令 - 确认满意度】\n"
        "你已经回答了用户的问题，现在需要确认用户是否满意：\n"
        "1. 主动询问'请问以上回答是否解决了您的问题？'\n"
        "2. 如果用户满意，询问是否还有其他问题\n"
        "3. 如果用户不满意，尝试换一种方式重新解答或建议转人工"
    ),
    ConversationStage.TRANSFERRING: (
        "【流程指令 - 转接人工】\n"
        "用户需要人工服务，你必须立即调用 transfer_to_human 工具。\n"
        "说明转接原因，并告知用户人工客服即将接入，在此之前不要再回答其他问题。"
    ),
    ConversationStage.ENDED: (
        "【流程指令 - 对话结束】\n"
        "对话即将结束，你需要：\n"
        "1. 发送结束语，感谢用户使用\n"
        "2. 简短询问用户对本次服务的满意度\n"
        "3. 不要再展开新的话题"
    ),
}

_SUMMARY_PROMPT = (
    "以下是本次对话的早期历史摘要：\n{summary}\n"
    "请基于以上上下文继续对话。"
)

_TURNS_EXCEEDED_MSG = (
    "本次对话已达到最大轮次限制，为了避免资源浪费，本次服务即将结束。\n"
    "如果您还有其他问题，欢迎重新发起对话。感谢您的使用！"
)


# ── 工具函数 ────────────────────────────────────────────────

def _classify_user_message(text: str) -> dict:
    text_lower = text.lower().strip()
    needs_human = any(kw in text_lower for kw in _HUMAN_REQUEST)
    is_negative = any(kw in text_lower for kw in _NEGATIVE)
    is_positive = any(kw in text_lower for kw in _POSITIVE)
    has_question = any(kw in text_lower for kw in _QUESTION_HINTS)
    return {
        "needs_human": needs_human,
        "is_negative": is_negative,
        "is_positive": is_positive,
        "has_question": has_question,
    }


def _get_thread_id() -> str:
    try:
        config = get_config()
        return config.get("configurable", {}).get("thread_id", "default")
    except Exception:
        return "default"


def _find_last_human(state: dict) -> str | None:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _find_last_ai(state: dict) -> str | None:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return None


# ── 消息窗口与摘要压缩 ──────────────────────────────────────

def _build_windowed_messages(state: dict, cs) -> list | None:
    """对消息列表做窗口裁剪 + 摘要压缩，返回替换后的 messages 或 None（不需要压缩）。"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 分离出对话消息（Human/AI），跳过 SystemMessage
    conversation_pairs: list[tuple] = []  # [(human, ai), ...]
    i = 0
    while i < len(messages):
        if isinstance(messages[i], HumanMessage):
            pair = [messages[i]]
            if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                pair.append(messages[i + 1])
                i += 2
            else:
                i += 1
            conversation_pairs.append(tuple(pair))
        else:
            i += 1

    # 不需要压缩
    if len(conversation_pairs) <= RECENT_WINDOW:
        return None

    # 需要压缩：旧消息生成摘要，保留最近窗口
    old_pairs = conversation_pairs[:-RECENT_WINDOW]
    recent_pairs = conversation_pairs[-RECENT_WINDOW:]

    # 构建摘要文本
    if cs.summary:
        summary_text = cs.summary
    else:
        summary_text = ""
    for pair in old_pairs:
        for msg in pair:
            if isinstance(msg, HumanMessage):
                summary_text += f"用户：{msg.content}\n"
            elif isinstance(msg, AIMessage):
                summary_text += f"客服：{msg.content}\n"

    cs.summary = summary_text

    # 组装新消息列表：摘要 SystemMessage + 最近窗口
    new_messages = [SystemMessage(content=_SUMMARY_PROMPT.format(summary=summary_text))]
    for pair in recent_pairs:
        new_messages.extend(pair)

    return new_messages


# ── 状态转移逻辑 ────────────────────────────────────────────

def _advance_stage(cs, analysis: dict) -> bool:
    """根据用户消息分析结果推进阶段，返回是否发生了变更。"""
    old = cs.stage

    # 超过最大轮次，强制结束
    if cs.turn_count >= MAX_TURNS:
        cs.transition_to(ConversationStage.ENDED)
        return cs.stage != old

    # 任何阶段：用户要求人工 → 立即转人工
    if analysis["needs_human"]:
        cs.needs_human = True
        cs.transition_to(ConversationStage.TRANSFERRING)
        return cs.stage != old

    if cs.stage == ConversationStage.GREETING:
        cs.transition_to(ConversationStage.INQUIRY)

    elif cs.stage == ConversationStage.INQUIRY:
        if analysis["has_question"] or analysis["is_negative"]:
            cs.transition_to(ConversationStage.ANSWERING)

    elif cs.stage == ConversationStage.ANSWERING:
        if analysis["is_positive"]:
            cs.unresolved_count = 0
            cs.transition_to(ConversationStage.CONFIRMING)
        elif analysis["is_negative"]:
            cs.unresolved_count += 1
            if cs.unresolved_count >= 3:
                cs.needs_human = True
                cs.transition_to(ConversationStage.TRANSFERRING)
            else:
                cs.transition_to(ConversationStage.INQUIRY)

    elif cs.stage == ConversationStage.CONFIRMING:
        if analysis["is_positive"]:
            cs.transition_to(ConversationStage.ENDED)
        elif analysis["is_negative"]:
            cs.transition_to(ConversationStage.INQUIRY)
        elif analysis["has_question"]:
            cs.transition_to(ConversationStage.ANSWERING)

    elif cs.stage == ConversationStage.TRANSFERRING:
        pass

    elif cs.stage == ConversationStage.ENDED:
        if analysis["has_question"]:
            cs.transition_to(ConversationStage.INQUIRY)
            cs.unresolved_count = 0

    return cs.stage != old


# ── Middleware ──────────────────────────────────────────────

@before_model(name="flow_pre")
def pre_handler(state, runtime) -> dict | None:
    """在 LLM 调用前：分析用户消息 → 推进状态 → 消息压缩 → 注入阶段指令。"""
    thread_id = _get_thread_id()
    cs = get_state(thread_id)

    last_human = _find_last_human(state)
    if last_human:
        cs.turn_count += 1
        analysis = _classify_user_message(last_human)
        _advance_stage(cs, analysis)

    updates = {}

    # 消息窗口压缩
    windowed = _build_windowed_messages(state, cs)
    if windowed is not None:
        updates["messages"] = windowed

    # 注入当前阶段专属指令
    instruction = _STAGE_INSTRUCTIONS.get(cs.stage)
    if instruction:
        updates.setdefault("messages", state.get("messages", []))
        updates["messages"] = list(updates["messages"]) + [SystemMessage(content=instruction)]

    # 超过最大轮次的兜底回复
    if cs.turn_count >= MAX_TURNS and cs.stage == ConversationStage.ENDED:
        updates["messages"] = [AIMessage(content=_TURNS_EXCEEDED_MSG)]

    return updates if updates else None


@after_model(name="flow_post")
def post_handler(state, runtime) -> dict | None:
    """在 LLM 回复后：补充确认提示或兜底消息。"""
    thread_id = _get_thread_id()
    cs = get_state(thread_id)

    if cs.stage == ConversationStage.TRANSFERRING:
        last_ai = _find_last_ai(state)
        if last_ai and "转接人工" not in last_ai and "转接" not in last_ai:
            return {"messages": [AIMessage(
                content="看起来我暂时无法很好地解决您的问题，正在为您转接人工客服，请稍候……"
            )]}

    if cs.stage == ConversationStage.ANSWERING and cs.turn_count % _CONFIRM_EVERY_N_TURNS == 0:
        last_ai = _find_last_ai(state)
        if last_ai and "是否解决" not in last_ai and "是否满意" not in last_ai:
            return {"messages": [AIMessage(
                content="\n请问以上回答是否解决了您的问题？"
            )]}

    return None
