from langchain_core.tools import tool


@tool
def transfer_to_human(reason: str) -> str:
    """当无法解决用户问题或用户明确要求时，转接人工客服。

    Args:
        reason: 需要转人工的原因
    """
    return (
        "正在为您转接人工客服，请稍候……\n"
        f"转接原因：{reason}\n"
        "人工客服会在1-2分钟内接入，感谢您的耐心等待。"
    )
