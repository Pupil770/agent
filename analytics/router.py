"""对话分析 REST 接口"""
from fastapi import APIRouter, Query

from analytics.db import (
    get_frequent_questions,
    get_rag_stats,
    get_satisfaction_stats,
    query_logs,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/logs")
async def list_logs(
    thread_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return {"logs": query_logs(thread_id, start_date, end_date, limit, offset)}


@router.get("/rag-stats")
async def rag_stats(
    start_date: str | None = None,
    end_date: str | None = None,
):
    return get_rag_stats(start_date, end_date)


@router.get("/frequent-questions")
async def frequent_questions(
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    return {"questions": get_frequent_questions(start_date, end_date, limit)}


@router.get("/satisfaction")
async def satisfaction_stats(
    start_date: str | None = None,
    end_date: str | None = None,
):
    return get_satisfaction_stats(start_date, end_date)
