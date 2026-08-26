# ADR-0005：流式 RAG API 与 SSE 事件契约

- 状态：Accepted
- 日期：2026-08-26

## 背景

M3-A 已在应用层完成问题、向量召回、TopK、安全 Prompt 和最终 Chat 模型流的显式编排，
但这些 Python 事件仍是内部类型。客户端需要一个稳定且可测试的公开入口，同时必须明确
响应开始前后的错误差异、事件顺序、请求追踪、超时、取消和资源释放语义。

当前尚未建立会话、消息或 RAG Run 表，也没有权限系统。公开端点因此只能接收显式知识库
作用域，不能隐式检索所有知识库，不能返回尚不存在的消息 ID，也不能声称已经保存会话。

## 决策

### 1. 端点与请求

在现有 `/api/v1` 下增加 `POST /chat/stream`。请求使用 JSON：

```json
{
  "question": "如何申请退款？",
  "scope": {
    "knowledge_base_ids": ["00000000-0000-0000-0000-000000000001"],
    "document_ids": [],
    "document_formats": [],
    "parser_names": [],
    "sections": [],
    "page_numbers": []
  }
}
```

`question` 去除首尾空白后必须为 1～10000 个字符。`knowledge_base_ids` 必须显式提供
1～100 个 UUID；其余过滤字段可省略，并沿用 `VectorSearchScope` 的去重、长度和正整数约束。
当前不接受 `conversation_id` 或 `user_id`，这些字段等会话持久化和权限边界建立后再加入。

服务端为每个合法请求生成 UUID，并通过 `X-Request-ID` 响应头以及每个事件的
`request_id` 字段返回。客户端不得用请求体指定该 ID。

### 2. 传输与事件框架

成功建立流时返回 HTTP 200 和 `text/event-stream`，并设置：

- `Cache-Control: no-cache, no-transform`
- `X-Accel-Buffering: no`

每个 SSE 帧固定包含 `id`、`event` 和单行 JSON `data`：

```text
id: <request_id>:<sequence>
event: <event_name>
data: {"request_id":"...","sequence":1,...}

```

`sequence` 从 1 开始严格递增；SSE `id` 由请求 ID 和序号组成。M3-B 不实现断线续传，
`Last-Event-ID` 不会触发重放。JSON 使用 UTF-8，换行等内容由 JSON 转义，避免破坏帧边界。

### 3. 公开事件 Schema

本版本定义五种事件：

1. `status`：增加 `stage`，当前可能为 `retrieving`、`prompting`、`generating`、
   `completed` 或 `no_context`。
2. `content`：增加非空 `delta`，只包含答案正文，不包含模型 reasoning。
3. `sources`：增加非空 `sources` 数组；每项包含连续 `citation_number`、Chunk/知识库/
   文档/版本 UUID、`source_key`、`display_name`、文档格式、可选章节和页码、内容哈希与
   Cosine 相似度，不包含文档正文。
4. `error`：增加稳定 `code`、可公开 `message` 和 `retryable`。
5. `done`：增加 `outcome`，取值为 `completed`、`no_context` 或 `error`；成功时可包含
   `model_id`、`finish_reason` 和 Token 用量。正常或空检索结局包含 Pipeline 返回的轻量
   `trace`；当前异常对象不携带部分上下文，因此 error 结局的 `trace` 为空。

正常回答的事件顺序为 status、零到多个 content、sources、completed status、done。
空检索为 retrieving status、no_context status、done，且不调用 Chat 模型。流开始后的失败
必须输出一个 error，紧接一个 `outcome=error` 的 done，然后关闭流；不得切换模型并重放已
输出的正文。客户端应忽略当前版本中不认识的事件名，以允许以后增加 `reply_to` 或
`guidance`，但改变现有字段语义、顺序保证或端点仍需更新 ADR 或 API 版本。

### 4. 错误、超时与取消

请求 Schema 无效或应用服务尚未就绪发生在流建立前，继续使用现有 JSON 错误外壳和 HTTP
422/503。Pipeline、检索或模型错误发生在流式响应建立后，HTTP 状态不能再修改，因此转换为
上述 error + done 事件；未知错误只返回通用 `internal_error`，不得公开异常文本、堆栈、
连接串、Prompt 或文档内容。

单一 `RAG_GLOBAL_TIMEOUT_SECONDS` 覆盖检索、Prompt 组装和模型流，默认 120 秒，并且不得小于
模型请求总超时。超时使用可重试的 `global_timeout` 流事件。

客户端断开或调用方提前关闭流时，取消继续向下传播，不伪造无法送达的 done 事件。API 必须
在 `finally` 中关闭 Pipeline 异步生成器；Pipeline 继续负责关闭模型流，供应商适配器继续
负责关闭底层 HTTP 流。

### 5. 组合与生命周期

默认 lifespan 在数据库和 Redis 打开后构建一个共享的最终 Chat 模型与 RAG Pipeline。
Chat HTTP 连接池由应用服务图拥有，并在数据库和 Redis 之前关闭。若 Chat 配置无效，启动
快速失败且已经打开的基础设施仍必须释放。M3-B 只组合最终回答模型，不提前实例化 fast、
Rerank、会话或后台任务能力。

## 备选方案

### 使用 WebSocket

WebSocket 适合双向交互，但当前问答是单请求、单方向增量输出。SSE 保留普通 HTTP 请求、代理
和浏览器事件流语义，复杂度更低。

### 使用 NDJSON 或裸文本块

实现较简单，但没有标准事件名、事件 ID 和类型边界，客户端难以可靠区分状态、正文、来源与
错误。

### 流开始前执行完整检索

这样部分检索错误可以返回 HTTP 4xx/5xx，但会延迟首个状态事件，并把 Pipeline 拆到路由层。
本项目保留单一显式 Pipeline，流内错误由稳定事件表达。

### 本阶段同时保存会话与 RAG Run

可以一次完成 M3，但会同时修改公开 API 和数据库核心表。为保持迁移、失败语义和测试范围
清晰，会话消息与最小 RAG Run 持久化放到下一子阶段并单独记录数据库决策。

## 后果

- 调用方可以通过一个稳定的 v1 SSE 协议消费进度、正文、来源、错误和完成状态。
- HTTP 200 只表示流已建立；最终成功与否必须以终端 done 事件判断。
- 当前没有认证、会话持久化、断线重放、心跳、限流或跨实例取消信号，只适合受控开发和演示。
- 默认 API 启动现在需要有效的 Chat API Key 配置；测试和真实数据库集成使用可控 Fake 模型，
  不访问云端模型或消耗额度。
- 后续加入会话消息 ID、澄清事件或持久化 Trace 时，必须保持本契约兼容或显式升级版本。
