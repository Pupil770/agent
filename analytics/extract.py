"""从 agent.invoke() 返回的消息列表中提取结构化数据"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def extract_turn_data(
    messages: list,
    thread_id: str,
    duration_ms: int,
    stage: str,
    turn_count: int,
) -> dict:
    # 最后一条用户消息
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 最后一条 AI 回复（优先取无 tool_calls 的最终回复）
    ai_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            ai_response = msg.content
            break
    if not ai_response:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break

    # 提取所有 tool_calls 和对应 tool_results
    # 先建 tool_call_id → name 的映射，用于关联 ToolMessage
    tc_name_map: dict[str, str] = {}
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "args": tc["args"]})
                tc_name_map[tc["id"]] = tc["name"]

    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "") or tc_name_map.get(msg.tool_call_id, "")
            tool_results.append({
                "name": name,
                "content": (msg.content or "")[:500],
            })

    # RAG 命中检测
    rag_used = any(tc["name"] == "knowledge_base_search" for tc in tool_calls)
    rag_has_result = False
    if rag_used:
        for tr in tool_results:
            if tr["name"] == "knowledge_base_search" and tr["content"]:
                rag_has_result = True
                break

    return {
        "thread_id": thread_id,
        "turn_count": turn_count,
        "user_message": user_message,
        "ai_response": ai_response,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "rag_used": rag_used,
        "rag_has_result": rag_has_result,
        "stage": stage,
        "duration_ms": duration_ms,
    }
