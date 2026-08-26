# ADR-0006：最小会话消息与 RAG Run 持久化

- 状态：Accepted
- 日期：2026-08-26

## 背景

M3-B 已提供公开流式问答 API，但每次请求结束后只在客户端保留正文、来源与 Trace。服务端
没有会话身份、消息事实或 RAG Run 结局，无法在 M4 安全加载最近消息，也无法区分完成、
空检索、模型失败和客户端取消。

本阶段只建立 M3 验收所需的最小持久化，不实现记忆窗口、摘要、Rewrite、Intent、Rerank、
会话列表或管理 CRUD。数据库与 SSE 都是公开/核心契约，必须先明确事务和失败语义。

## 决策

### 1. 数据表

新增三个 PostgreSQL 表：

1. `conversations`：只保存 UUID、创建时间和更新时间。当前没有用户或租户字段，因为项目尚未
   建立可信身份边界。
2. `rag_runs`：保存唯一 `request_id`、会话 ID、显式知识库 ID 数组、运行状态、模型 ID、
   finish reason、Token 用量、阶段 Trace、引用 Chunk ID、稳定错误码和起止时间。不保存
   Prompt、推理内容或完整检索正文。
3. `messages`：保存会话 ID、可空 RAG Run ID、严格递增的会话内 `ordinal`、`user` 或
   `assistant` 角色、非空正文和创建时间。一个 Run 最多各有一条 user/assistant 消息。

`conversation_summaries` 和独立 `rag_trace_nodes` 留到 M4/评测需要明确后再建立；当前轻量 Trace
使用 `rag_runs.trace` JSONB，结构仍由类型化领域模型生成。

删除会话时级联删除它的 Run 与消息。删除单个 Run 时消息保留但解除 `run_id`，避免把消息事实
错误地当成调试记录。当前不提供公开删除端点。

### 2. 状态与事务边界

Run 状态固定为：

- `running`：用户消息与 Run 已提交，Pipeline 尚未产生持久化终局。
- `completed`：答案、来源、模型结果和 Trace 已原子保存，同时新增 assistant 消息。
- `no_context`：空检索正常短路；保存 Trace，但不创建客户端从未收到的 assistant 消息。
- `failed`：检索、模型、协议、持久化或未知错误；只保存稳定错误码。
- `cancelled`：客户端断开、任务取消或调用方提前关闭流。

开始事务原子创建或锁定会话、创建 running Run、分配 user 消息 ordinal 并保存用户问题。完成
事务锁定会话与 Run，必要时新增 assistant 消息，再一次性写入终局字段。错误与取消使用独立
短事务把仍为 running 的 Run 转为终态。

成功 `done` 事件必须在完成事务提交后才发送，因此客户端收到 completed/no_context done 时，
对应事实已经持久化。若完成持久化失败，不发送成功 done，而是按流内错误返回。

### 3. 会话并发与消息顺序

首次请求省略 `conversation_id`，服务端生成新会话。后续请求可以携带已有 UUID；不存在的会话
返回稳定 `conversation_not_found` 流错误，服务端不会接受客户端指定 UUID 创建新会话。

P0 暂不允许同一会话同时存在两个 running Run。应用在会话行锁内检查，数据库使用部分唯一
索引兜底；冲突返回可重试 `conversation_busy`。这避免并发回答导致 user/assistant ordinal
交错，在 M4 加载历史时产生错误对话顺序。不同会话仍可并发。

### 4. Pipeline 组合与失败语义

使用 `PersistentStreamingRagPipeline` 装饰现有 `BasicStreamingRagPipeline`：

- 内部基础 Pipeline 继续只负责 Retrieval、Prompt 和模型流，不依赖 SQLAlchemy。
- 装饰器通过领域 `RagRunRepository` 端口开始 Run、收集已公开的正文/来源/Trace 并写入终局。
- 装饰器在调用方提前关闭、`CancelledError` 或 `GeneratorExit` 时标记 cancelled，并始终关闭
  内层异步生成器。
- 已经输出的部分正文在失败或取消时不保存为 assistant 消息，避免未来记忆把不完整回答当作
  正常事实；Run 状态仍保留失败/取消证据。
- SQLAlchemy 异常转换为脱敏 `persistence_failure`，不公开 SQL、DSN 或消息正文。

默认运行时注入 SQLAlchemy 仓储；单元测试可以使用内存 Fake，不需要数据库。

### 5. API 与 SSE 增量契约

`POST /api/v1/chat/stream` 请求新增可选 `conversation_id`。省略表示创建新会话，提供表示继续
已有会话。新增规划中的 `reply_to` 事件，且它是成功开始 Run 后的第一个事件：

```json
{
  "request_id": "...",
  "sequence": 1,
  "conversation_id": "...",
  "user_message_id": "...",
  "rag_run_id": "..."
}
```

客户端用 `conversation_id` 发起后续轮次；`user_message_id` 明确当前回答对应的用户消息；
`rag_run_id` 用于未来评测和诊断。其余事件继续遵守 ADR-0005，序号在 reply_to 后递增。

HTTP 200 仍只表示 SSE 流建立。会话不存在、会话忙或开始持久化失败发生在生成器首次执行时，
因此使用 error + done 流事件；请求 Schema 无效仍使用流开始前的 HTTP 422。

### 6. 当前不会做的事

- 不从历史消息构建 Prompt，也不声称已经实现 Memory。
- 不建立用户所有权、认证、多租户或会话分享边界。
- 不提供会话/消息查询、编辑、删除或分页 API。
- 不保存 reasoning、Prompt、完整召回正文或异常详情。
- 不使用 Redis 作为会话唯一事实来源，也不引入任务队列。

## 备选方案

### 在现有 Pipeline 中直接写 SQLAlchemy

代码更少，但会破坏应用层对数据库实现的隔离，也让基础 Pipeline 单元测试依赖数据库。

### 流结束后由 API 路由统一保存

路由会承担正文聚合、状态机和取消处理，并且容易在客户端断开时漏写终态。装饰器更贴合
Pipeline 生命周期，同时保持 SSE 适配器只负责传输映射。

### 允许同一会话并发 Run

吞吐更高，但 assistant 完成顺序可能与 user 提交顺序不同。当前没有分支对话或 parent message
语义，先串行化同一会话更可解释。

### 保存每个 Trace 节点为独立表

便于复杂查询，但当前只有三个阶段且 Trace 较小。先使用结构化 JSONB，等评测查询证明需要后
再规范化，避免过早建立大量表和迁移成本。

## 后果

- 每个真正开始的问答请求都有可追溯 user 消息和 RAG Run 终局。
- completed 回答可以在 PostgreSQL 中还原消息正文、来源 Chunk、模型和阶段耗时；空检索、错误
  与取消不会伪装成成功 assistant 消息。
- SSE 新增 reply_to，请求新增 `conversation_id`，属于 ADR-0005 的兼容增量；旧客户端按约定
  忽略未知事件仍可消费正文和 done。
- 当前会话只是受控开发环境中的技术身份，不是访问控制边界。M4 使用历史前必须继续设计窗口、
  摘要、截断与 Prompt 注入防护。
