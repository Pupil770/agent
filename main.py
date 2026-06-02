import os
import time
from contextlib import asynccontextmanager

from config.apikey import DASHSCOPE_API_KEY
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.sqlite import SqliteSaver

from analytics import analytics_router, extract_turn_data, init_db, log_turn
from flow import pre_handler, post_handler, transfer_to_human
from flow.state import get_state
from rag import get_vectorstore, knowledge_router
from tools import calculate, get_weather

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

model = ChatTongyi(model="qwen3-max")

SYSTEM_PROMPT = (
    "你是一个智能客服助手。请遵循以下流程：\n"
    "1. 首次对话时主动打招呼并询问需要什么帮助\n"
    "2. 优先使用知识库搜索工具回答企业相关问题\n"
    "3. 回答后关注用户是否满意\n"
    "4. 如果连续无法解决用户问题，主动建议转接人工\n"
    "5. 用户要求转人工时，调用 transfer_to_human 工具，并说明转接原因\n"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_tool = create_retriever_tool(
        retriever,
        name="knowledge_base_search",
        description="搜索企业知识库以回答客户问题。当用户询问产品、服务、政策或常见问题时使用此工具。",
    )

    with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
        app.state.agent = create_agent(
            model=model,
            tools=[get_weather, calculate, rag_tool, transfer_to_human],
            system_prompt=SYSTEM_PROMPT,
            middleware=[pre_handler, post_handler],
            checkpointer=checkpointer,
        )
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(knowledge_router)
app.include_router(analytics_router)


@app.post("/chat")
async def chat(message: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    res = app.state.agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    cs = get_state(thread_id)
    turn_data = extract_turn_data(
        messages=res["messages"],
        thread_id=thread_id,
        duration_ms=duration_ms,
        stage=cs.stage.value,
        turn_count=cs.turn_count,
    )
    log_turn(**turn_data)

    return {"response": res["messages"][-1].content}


@app.get("/conversation/{thread_id}")
async def conversation_state(thread_id: str):
    cs = get_state(thread_id)
    return cs.to_dict()
