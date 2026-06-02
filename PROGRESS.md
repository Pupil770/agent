# 项目进度：企业问答数字分身

## 项目目标

基于 FastAPI + LangChain + Qwen 构建企业智能客服，具备知识库检索、对话流程管控、人工接管等能力。

## 已完成模块

### 1. Agent 基础框架

- **文件**: `main.py`, `tools.py`, `config/apikey.py`
- **能力**: FastAPI 服务 + LangGraph Agent + ChatTongyi(qwen3-max)
- **接口**: `POST /chat?message=&thread_id=`
- **对话记忆**: SqliteSaver 按 thread_id 持久化

### 2. RAG 知识库

- **文件**: `rag/store.py`, `rag/loader.py`, `rag/ingestion.py`, `rag/router.py`
- **能力**: 文档上传入库、语义检索、知识库管理
- **支持格式**: PDF, Word, TXT, Markdown, CSV, XLSX（FAQ 自动识别问答列）
- **向量存储**: ChromaDB 本地持久化（`knowledge_data/`）
- **嵌入模型**: DashScope text-embedding-v3
- **接口**:
  - `POST /knowledge/upload` — 上传文档
  - `GET /knowledge/documents` — 列出文档
  - `DELETE /knowledge/documents/{doc_id}` — 删除文档
  - `POST /knowledge/query` — 调试查询
- **后端可切换**: 改 `rag/store.py` 即可迁移 pgvector，业务代码零改动

### 3. 对话流程管控（已完成）

- **文件**: `flow/state.py`, `flow/middleware.py`, `flow/tools.py`
- **能力**:
  - 6 阶段状态机定义（greeting / inquiry / answering / confirming / transferring / ended）
  - **状态驱动决策**：每个阶段注入专属 SystemMessage 指令，强制引导 LLM 行为
  - Pre-middleware：轮次计数、意图关键词分类、不满计数、状态转移
  - Post-middleware：连续未解决 ≥3 次自动建议转人工、每 N 轮追加满意度确认
  - `transfer_to_human` 工具：LLM 可主动调用
  - `GET /conversation/{thread_id}` 查询会话状态

### 4. 多轮对话优化（已完成）

- **文件**: `flow/middleware.py`（`_build_windowed_messages`），`flow/state.py`（`summary` 字段）
- **能力**:
  - **消息窗口压缩**：超过 `RECENT_WINDOW`（6轮）时，将旧对话压缩为摘要 SystemMessage
  - **摘要累积**：`ConversationState.summary` 字段保存早期对话摘要，压缩后不再丢失上下文
  - **轮次限制**：超过 `MAX_TURNS`（30轮）自动结束对话，返回兜底消息
  - **注入消息不累积**：每次注入的阶段指令是单条 SystemMessage，旧消息通过窗口机制自然淘汰

### 6. 对话日志与分析（已完成）

- **文件**: `analytics/db.py`, `analytics/extract.py`, `analytics/router.py`
- **能力**:
  - **对话日志记录**：每轮完整交互写入 SQLite（`analytics.db`），包括用户消息、AI 回复、工具调用及结果、RAG 使用情况、阶段、耗时
  - **知识库命中率统计**：通过 `rag_used` / `rag_has_result` 布尔标志，聚合计算 RAG 调用率和命中率
  - **高频问题挖掘**：按 `user_message` 分组统计频次，附带 RAG 命中情况，识别知识库盲区
  - **满意度分析**：基于对话阶段分布代理，`ended_rate` 为满意代理，`transfer_rate` 为不满意代理
- **接口**:
  - `GET /analytics/logs` — 查询对话日志（支持 thread_id/日期过滤）
  - `GET /analytics/rag-stats` — 知识库命中率统计
  - `GET /analytics/frequent-questions` — 高频问题挖掘
  - `GET /analytics/satisfaction` — 满意度分析

---

## 设计详解

### 一、状态机架构

#### 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                     用户消息到达                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  @before_model (pre_handler)                                │
│  1. 分析用户消息（关键词分类）                                │
│  2. 根据分析结果推进状态（_advance_stage）                   │
│  3. 消息窗口压缩（超出阈值时摘要旧消息）                     │
│  4. 注入当前阶段专属 SystemMessage 指令                      │
│  5. 超轮次兜底                                               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM 生成回复                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  @after_model (post_handler)                                │
│  1. 转人工兜底：确保调用了 transfer_to_human                 │
│  2. 每 N 轮追加满意度确认                                    │
└─────────────────────────────────────────────────────────────┘
```

#### 六阶段状态机

| 阶段 | 触发条件 | 注入指令 | 状态转移 |
|------|---------|---------|---------|
| **GREETING** | 新会话初始化 | 强制先问候再询问需求 | 用户首次发言 → INQUIRY |
| **INQUIRY** | 用户意图不明确 | 引导描述问题、追问细节 | 用户提出问题 → ANSWERING |
| **ANSWERING** | 用户问题明确 | 优先知识库检索、不编造 | 用户满意 → CONFIRMING；不满意 → INQUIRY |
| **CONFIRMING** | 回答完成 | 主动询问是否解决 | 满意 → ENDED；不满意 → INQUIRY；新问题 → ANSWERING |
| **TRANSFERRING** | 用户要求人工 / 连续3次未解决 | 必须调用 transfer_to_human | 不自动转移 |
| **ENDED** | 用户确认满意 / 超过最大轮次 | 发送结束语、询问满意度 | 用户新问题 → INQUIRY（重新开始） |

#### 关键词分类

```python
_POSITIVE = {"好", "好的", "谢谢", "解决了", "可以了", "明白了", ...}
_NEGATIVE = {"没有", "不行", "没解决", "没帮助", "不明白", "投诉", ...}
_HUMAN_REQUEST = {"转人工", "人工客服", "真人", "找人工", ...}
_QUESTION_HINTS = {"吗", "怎么", "如何", "什么", "为什么", "帮我", "请问", ...}
```

#### 状态转移图

```
                    ┌──────────────┐
                    │   GREETING   │
                    └──────┬───────┘
                           │ 用户发言
                           ▼
                    ┌──────────────┐
              ┌─────│   INQUIRY    │◄────────┐
              │     └──────┬───────┘         │
              │            │ 提出问题        │ 不满意
              │            ▼                 │
              │     ┌──────────────┐         │
              │     │  ANSWERING   │─────────┘
              │     └──────┬───────┘
              │            │ 满意
              │            ▼
              │     ┌──────────────┐
              │     │ CONFIRMING   │───────┐
              │     └──────┬───────┘       │ 新问题
              │            │ 满意          │
              │            ▼               ▼
              │     ┌──────────────┐  ┌──────────────┐
              │     │    ENDED     │  │  ANSWERING   │
              │     └──────────────┘  └──────────────┘
              │            ▲
              │            │ 新问题
              └────────────┘

    任意阶段 ──"转人工"──► ┌──────────────┐
                          │ TRANSFERRING │
                          └──────────────┘

    超过 MAX_TURNS ──────► ┌──────────────┐
                          │    ENDED     │
                          └──────────────┘
```

### 二、多轮对话优化

#### 消息窗口压缩机制

核心问题：随着对话增长，消息列表不断膨胀，最终超出模型 token 限制。

解决方案：滑动窗口 + 摘要压缩。

```
原始消息列表（假设对话了 10 轮 = 20 条消息）：
┌──────────────────────────┐
│ Human: 你好               │
│ AI: 您好！请问...         │  ← 旧消息（将被压缩）
│ Human: 退货政策是什么      │
│ AI: 我们的退货政策...      │
│ ...                       │
│ Human: 还能换货吗          │
│ AI: 是的，可以...          │  ← 最近 6 轮（完整保留）
│ Human: 运费谁出           │
│ AI: 运费由...             │
└──────────────────────────┘

压缩后：
┌──────────────────────────────────────────┐
│ System: [摘要] 用户：你好                  │
│         客服：您好！请问...                │
│         用户：退货政策是什么               │  ← 摘要 SystemMessage
│         客服：我们的退货政策...            │
│         ...                               │
│ Human: 还能换货吗                          │
│ AI: 是的，可以...                          │  ← 最近 6 轮完整保留
│ Human: 运费谁出                           │
│ AI: 运费由...                             │
└──────────────────────────────────────────┘
```

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `MAX_TURNS` | 30 | 最大对话轮次，超过自动结束 |
| `RECENT_WINDOW` | 6 | 保留最近 N 轮完整消息 |
| `_CONFIRM_EVERY_N_TURNS` | 3 | 每隔几轮主动确认满意度 |

#### 压缩流程

```
pre_handler 被调用
    │
    ├── 1. 分析用户消息、推进状态
    │
    ├── 2. _build_windowed_messages()
    │       ├── 统计对话轮数（Human/AI 配对）
    │       ├── 轮数 <= RECENT_WINDOW？→ 不压缩，返回 None
    │       ├── 轮数 > RECENT_WINDOW？→ 拆分为 旧消息 + 最近窗口
    │       ├── 将旧消息追加到 ConversationState.summary（累积式）
    │       └── 返回 [摘要SystemMessage] + 最近窗口消息
    │
    ├── 3. 注入阶段指令 SystemMessage
    │
    └── 4. 超轮次兜底（MAX_TURNS）
```

#### 轮次限制

当 `turn_count >= MAX_TURNS` 时：
1. `_advance_stage` 强制将状态转移到 ENDED
2. `pre_handler` 直接返回一条固定的结束 AIMessage，跳过 LLM 调用
3. 用户再次发送消息时，如果包含问题关键词，会从 ENDED 重新开始到 INQUIRY

### 三、对话日志与分析

#### 架构

日志记录发生在 `/chat` 端点——这是唯一能获取完整一轮交互数据（用户消息、工具调用链、AI 回复、耗时）的地方。Middleware 只在单次 LLM 调用前后运行，一轮带工具调用的对话可能触发多次 middleware，不适合做轮次级日志。

```
/chat 端点
    │
    ├── time.perf_counter() → t0
    ├── agent.invoke() → res（完整消息列表）
    ├── time.perf_counter() → t1，计算 duration_ms
    │
    ├── extract_turn_data(res["messages"])
    │       ├── 提取最后一条 HumanMessage → user_message
    │       ├── 提取最后一条无 tool_calls 的 AIMessage → ai_response
    │       ├── 遍历所有 AIMessage.tool_calls → tool_calls[]
    │       ├── 遍历所有 ToolMessage → tool_results[]
    │       └── 检测 knowledge_base_search 调用 → rag_used, rag_has_result
    │
    └── log_turn(...) → 写入 analytics.db
```

#### 表结构

```sql
conversation_logs:
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  thread_id       TEXT NOT NULL
  turn_count     INTEGER NOT NULL
  user_message    TEXT NOT NULL
  ai_response     TEXT NOT NULL
  tool_calls      TEXT DEFAULT '[]'    -- JSON: [{name, args}]
  tool_results    TEXT DEFAULT '[]'    -- JSON: [{name, content(截断500字)}]
  rag_used        INTEGER DEFAULT 0    -- 是否调用了 RAG
  rag_has_result  INTEGER DEFAULT 0    -- RAG 是否返回了结果
  stage           TEXT DEFAULT ''      -- 对话阶段
  duration_ms     INTEGER DEFAULT 0    -- 耗时
  created_at      TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
```

关键设计：
- `rag_used` / `rag_has_result` 为布尔标志，命中率统计无需解析 JSON
- `tool_results` 内容截断 500 字符，完整数据在 LangGraph checkpointer 中
- 数据库文件 `analytics.db` 与 `chat_history.db` 分离（关注点分离：checkpointer 管对话状态，analytics 管日志）

#### RAG 命中检测

当 LLM 决定使用知识库时，LangGraph 的消息序列为：

```
HumanMessage(content="退货政策是什么")
  → AIMessage(tool_calls=[{name: "knowledge_base_search", args: {query: "退货政策"}}])
  → ToolMessage(name="knowledge_base_search", content="检索到的文档...")
  → AIMessage(content="基于知识库的回答...")
```

`extract_turn_data()` 扫描消息列表：
- 发现 `AIMessage.tool_calls` 中 `name=="knowledge_base_search"` → `rag_used=True`
- 通过 `tool_call_id` 关联到对应 `ToolMessage`，检查 content 非空 → `rag_has_result=True`
- `rag_used=True` 且 `rag_has_result=False` → RAG 被调用但未命中

#### 分析接口

| 端点 | 用途 | 核心查询 |
|------|------|---------|
| `GET /analytics/logs` | 查询原始日志 | `SELECT * FROM conversation_logs WHERE ...` |
| `GET /analytics/rag-stats` | RAG 命中率 | `SUM(rag_used) / COUNT(*)` = 调用率，`SUM(rag_has_result) / SUM(rag_used)` = 命中率 |
| `GET /analytics/frequent-questions` | 高频问题 | `GROUP BY user_message ORDER BY COUNT(*) DESC`，附带 RAG 命中列识别知识库盲区 |
| `GET /analytics/satisfaction` | 满意度 | `GROUP BY stage`，`ended_rate` = 正常结束占比，`transfer_rate` = 转人工占比 |

#### 高频问题 → 知识库反馈循环

运营工作流（非自动化代码）：

1. 查询 `GET /analytics/frequent-questions`
2. 识别 `count` 高但 `rag_hit_count` 为 0 的问题
3. 这些是知识库未能回答的高频问题
4. 针对这些主题创建文档，上传到 `POST /knowledge/upload`
5. 下次同类问题即可命中 RAG

#### 满意度代理方法

项目没有显式用户反馈机制，使用对话阶段分布作为满意度代理：

- **ended 阶段**：对话正常结束（用户满意或确认）→ 满意代理
- **transferring 阶段**：对话转人工（未解决）→ 不满意代理
- **answering/inquiry 阶段**：对话在这些阶段结束（可能未解决）

如后续增加显式反馈（如评价按钮），可向表添加 `satisfaction_score` 列。

---

### 四、设计优劣总结

#### 优势

| 设计点 | 说明 |
|--------|------|
| **确定性流程控制** | 阶段指令强制注入，不依赖 LLM "自觉"遵循 |
| **摘要不丢上下文** | 旧消息压缩为摘要而非直接截断，LLM 仍能感知早期对话 |
| **累积式摘要** | `summary` 字段持续累积，多次压缩不会丢失信息 |
| **可配置参数** | 窗口大小、轮次限制、确认间隔均可调 |
| **可观测性** | `last_stage`、`summary_length`、`turn_count` 便于调试 |
| **线程安全** | `ConversationState` 使用 `threading.Lock` |
| **兜底机制完备** | 转人工兜底、满意度确认兜底、超轮次兜底 |
| **日志与分析分离** | `analytics.db` 与 `chat_history.db` 各司其职，互不干扰 |
| **布尔标志加速查询** | `rag_used`/`rag_has_result` 避免解析 JSON 即可统计命中率 |
| **高频问题直通知识库** | 分析→识别盲区→补充文档，闭环运营无需改代码 |

#### 局限与权衡

| 局限 | 原因 | 缓解方案 |
|------|------|---------|
| **摘要未用 LLM 压缩** | 当前摘要只是原文拼接，未做语义压缩 | 可用 LLM 生成精简摘要，减少 token 消耗 |
| **关键词分类粗糙** | 简单集合匹配，无法理解语义 | 可替换为小模型意图分类器 |
| **状态存储内存级** | `_conversations` 是进程内字典，重启丢失 | 可迁移到 Redis/数据库 |
| **窗口基于轮数而非 token** | 可能窗口内消息本身就超长 | 可加 token 计数辅助判断 |
| **摘要与指令共享 SystemMessage 通道** | LLM 可能混淆摘要上下文和流程指令 | 可区分摘要前缀和指令前缀 |
| **无超时机制** | 阶段没有超时自动推进 | 可增加定时器触发 TRANSFERRING |
| **满意度为代理指标** | 无显式用户反馈，用阶段分布近似 | 后续可加评价按钮 + `satisfaction_score` 列 |
| **高频问题按原文分组** | 同义不同表述会分为多条 | 可用 LLM 做语义聚类，或用 embedding 相似度合并 |

---

## 未完成部分

### 3.2 转人工后续逻辑

`transfer_to_human` 只返回提示文字，无实际接管能力：

- **会话移交**: 将对话历史打包，通知人工客服系统
- **人工接入**: 客服接单后接管 thread，agent 暂停响应
- **会话回传**: 人工处理结果写回对话记录，agent 后续可感知

依赖外部客服渠道（企业微信/钉钉/工单系统），需先确定对接目标。

### 5. 多渠道接入层

当前只有 HTTP API，实际需对接：

- 网页客服窗口（WebSocket）
- 微信公众号 / 企业微信
- 钉钉机器人

需要一个适配层将各渠道消息统一转换为内部格式。

### 7. 性能与安全

- 请求限流（防滥用）
- 输入过滤（防 prompt 注入）
- 输出审核（敏感信息检测）
- 超时兜底回复