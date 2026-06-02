"""对话日志数据库：建表、插入、查询"""
import json
import os
import sqlite3

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytics.db")
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id     TEXT NOT NULL,
            turn_count    INTEGER NOT NULL,
            user_message  TEXT NOT NULL,
            ai_response   TEXT NOT NULL,
            tool_calls    TEXT NOT NULL DEFAULT '[]',
            tool_results  TEXT NOT NULL DEFAULT '[]',
            rag_used      INTEGER NOT NULL DEFAULT 0,
            rag_has_result INTEGER NOT NULL DEFAULT 0,
            stage         TEXT NOT NULL DEFAULT '',
            duration_ms   INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_logs_thread ON conversation_logs(thread_id);
        CREATE INDEX IF NOT EXISTS idx_logs_created ON conversation_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_rag ON conversation_logs(rag_used);
    """)


def log_turn(
    thread_id: str,
    turn_count: int,
    user_message: str,
    ai_response: str,
    tool_calls: list[dict],
    tool_results: list[dict],
    rag_used: bool,
    rag_has_result: bool,
    stage: str,
    duration_ms: int,
):
    conn = get_conn()
    conn.execute(
        """INSERT INTO conversation_logs
           (thread_id, turn_count, user_message, ai_response, tool_calls,
            tool_results, rag_used, rag_has_result, stage, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (thread_id, turn_count, user_message, ai_response,
         json.dumps(tool_calls, ensure_ascii=False),
         json.dumps(tool_results, ensure_ascii=False),
         int(rag_used), int(rag_has_result), stage, duration_ms),
    )
    conn.commit()


def query_logs(
    thread_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    conn = get_conn()
    clauses, params = [], []
    if thread_id:
        clauses.append("thread_id = ?")
        params.append(thread_id)
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM conversation_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [dict(r) for r in rows]


def get_rag_stats(start_date: str | None = None, end_date: str | None = None) -> dict:
    conn = get_conn()
    clauses, params = [], []
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(
        f"""SELECT COUNT(*) as total_turns,
                   SUM(rag_used) as rag_called,
                   SUM(rag_has_result) as rag_hit
            FROM conversation_logs {where}""",
        params,
    ).fetchone()
    total = row["total_turns"] or 0
    called = row["rag_called"] or 0
    hit = row["rag_hit"] or 0
    return {
        "total_turns": total,
        "rag_called": called,
        "rag_hit": hit,
        "rag_hit_rate": round(hit / called, 4) if called else 0,
        "rag_call_rate": round(called / total, 4) if total else 0,
    }


def get_frequent_questions(
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = get_conn()
    clauses, params = [], []
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""SELECT user_message, COUNT(*) as count,
                   SUM(rag_used) as rag_used_count,
                   SUM(rag_has_result) as rag_hit_count
            FROM conversation_logs {where}
            GROUP BY user_message
            ORDER BY count DESC
            LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def get_satisfaction_stats(start_date: str | None = None, end_date: str | None = None) -> dict:
    conn = get_conn()
    clauses = ["stage != ''"]
    params = []
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(
        f"""SELECT stage, COUNT(*) as count
            FROM conversation_logs {where}
            GROUP BY stage
            ORDER BY count DESC""",
        params,
    ).fetchall()
    stage_dist = {r["stage"]: r["count"] for r in rows}
    total = sum(stage_dist.values())
    ended = stage_dist.get("ended", 0)
    transferring = stage_dist.get("transferring", 0)
    return {
        "stage_distribution": stage_dist,
        "total_turns": total,
        "ended_rate": round(ended / total, 4) if total else 0,
        "transfer_rate": round(transferring / total, 4) if total else 0,
    }
