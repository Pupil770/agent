from langchain.agents.middleware import after_model, before_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config

from flow.state import ConversationStage, get_state

# 满意/不满意关键词
_POSITIVE = {"好", "好的", "谢谢", "感谢", "没问题", "解决了", "可以了", "明白了", "懂了", "是的"}
_NEGATIVE = {"没有", "不是", "不行", "不行了", "还是不行", "没解决", "没帮助", "不明白", "不懂",
             "不合理", "差", "差评", "投诉", "人工", "转人工", "客服", "不满意"}
_HUMAN_REQUEST = {"转人工", "人工客服", "人工服务", "真人", "接线员", "找人工", "叫人工", "联系人工"}


def _classify_user_message(text: str) -> dict:
    text_lower = text.lower().strip()
    needs_human = any(kw in text_lower for kw in _HUMAN_REQUEST)
    is_negative = any(kw in text_lower for kw in _NEGATIVE)
    is_positive = any(kw in text_lower for kw in _POSITIVE)
    return {
        "needs_human": needs_human,
        "is_negative": is_negative,
        "is_positive": is_positive,
    }


def _get_thread_id() -> str:
    try:
        config = get_config()
        return config.get("configurable", {}).get("thread_id", "default")
    except Exception:
        return "default"


@before_model(name="flow_pre")
def pre_handler(state, runtime) -> dict | None:
    thread_id = _get_thread_id()
    cs = get_state(thread_id)

    # 找到最后一条用户消息
    last_human = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if last_human:
        cs.turn_count += 1
        analysis = _classify_user_message(last_human)
        if analysis["needs_human"]:
            cs.needs_human = True
        if analysis["is_negative"]:
            cs.unresolved_count += 1
        elif analysis["is_positive"]:
            cs.unresolved_count = 0

        if cs.stage == ConversationStage.GREETING:
            cs.stage = ConversationStage.INQUIRY
        elif cs.stage == ConversationStage.ANSWERING:
            cs.stage = ConversationStage.INQUIRY

    return None


@after_model(name="flow_post")
def post_handler(state, runtime) -> dict | None:
    thread_id = _get_thread_id()
    cs = get_state(thread_id)

    # 如果已标记需要转人工，追加提示
    if cs.needs_human:
        cs.stage = ConversationStage.TRANSFERRING
        return None

    # 连续未解决 >= 3 次，追加人工建议
    if cs.unresolved_count >= 3:
        cs.needs_human = True
        cs.stage = ConversationStage.TRANSFERRING
        last_ai = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai and "转接人工" not in last_ai.content:
            suffix = "\n\n看起来我暂时无法很好地解决您的问题，是否需要为您转接人工客服？"
            return {"messages": [AIMessage(content=suffix)]}

    cs.stage = ConversationStage.ANSWERING
    return None
