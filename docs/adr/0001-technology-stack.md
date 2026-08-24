# ADR-0001：首版技术栈与编排方式

- 状态：已接受
- 日期：2026-08-24
- 决策范围：P0 首个可验证版本

## 背景

Customer Agent 2 需要把一个复杂 Java RAG 参考项目中的大模型应用能力重新设计为 Python 项目。主链路包含问题改写、意图、检索预算、并行通道、RRF、Rerank、引用、SSE、取消和模型降级，不适合只用一个高层 Chain 隐藏全部状态。

项目同时面向开源和简历展示，因此除了开发速度，还必须考虑：

- 代码是否容易阅读和解释。
- 失败路径是否可以测试。
- 外部服务是否可以替换。
- 评测是否可以控制单一变量。
- 新用户是否可以通过 Docker 在本地复现。

## 决策

### Web 与异步运行时

- 使用 FastAPI 提供 HTTP API。
- 使用 Pydantic 定义请求、响应和配置。
- 使用 Uvicorn 运行 ASGI 应用。
- 使用 asyncio/AnyIO 表达结构化并发、超时和取消。

### 业务编排

- 使用项目自有的显式异步 Pipeline。
- 每个阶段具有明确输入、输出和 Trace。
- 不以 LangChain 或 LlamaIndex 作为主流程骨架。
- P0 不引入 LangGraph；只有实现持久化 ReAct Agent 时重新评估。

### 数据与向量

- 使用 PostgreSQL 保存业务数据。
- 使用 pgvector 保存和检索向量。
- 使用 SQLAlchemy 2 Async ORM 和 asyncpg。
- 使用 Alembic 管理数据库迁移。
- 使用 Redis 处理限流、任务状态、取消信号和可选缓存。

### 模型调用

- 使用 OpenAI-compatible 异步客户端调用阿里云 Chat 模型。
- 特殊端点使用 httpx 和独立适配器。
- final、fast、embedding、rerank 和 VLM 分别配置。
- 默认本地 Embedding Baseline 为 `BAAI/bge-base-zh-v1.5`。
- 默认专用 Rerank 为 `qwen3-rerank`，未配置时使用可观察的 No-op 降级。

### 文档与对象存储

- P0 解析 Markdown、TXT、PDF、DOCX 和 CSV。
- 文档解析器采用注册表与统一接口。
- 原始文件通过 S3-compatible 接口抽象；开发初期可用文件系统适配器。

### 测试与质量

- 使用 pytest、pytest-asyncio。
- 外部 HTTP 使用 respx 或等价 Mock。
- PostgreSQL/Redis 集成测试优先使用 Testcontainers。
- 使用 Ruff 做格式和静态规则检查，使用 Pyright 做类型检查。
- 使用固定评测集执行 Intent、Retrieval、Rerank 和延迟评测。

## 原因

### 为什么选择 FastAPI

- 与 Python 类型注解和 Pydantic 配合自然。
- ASGI 适合 LLM 流式输出和客户端取消。
- 自动生成 OpenAPI，减少重复接口文档。
- 生态成熟，便于开源用户理解和运行。

### 为什么首版选择 PostgreSQL + pgvector

- 业务数据和向量数据可以使用同一事务与权限边界。
- 当前数据规模适合单库方案，不需要提前引入 Milvus 运维复杂度。
- SQL 过滤便于落实知识库和文档作用域，而不是结果返回后再过滤。
- 后续仍可通过 `VectorStore` 接口增加 Milvus 适配器。

### 为什么不用 LangChain 作为主骨架

- 项目需要展示明确的阶段、短路和降级语义。
- 多通道检索、三段预算、引用编号和流取消需要直接控制。
- 高层抽象升级可能引入行为变化，降低源码可解释性。
- 不排斥使用独立工具库，但核心业务状态必须由项目自身掌握。

### 为什么区分 final 和 fast 模型

- 最终回答更关注综合质量。
- Rewrite、Intent、Summary 等内部任务更关注稳定结构、低延迟和低成本。
- 单一大模型承担所有任务会放大额度不足和延迟问题。

## 后果

### 正面影响

- 主流程可逐阶段调试和测试。
- 模型或向量存储替换不侵入业务 Pipeline。
- 评测可以控制变量并定位失败阶段。
- 项目结构适合面试时解释设计与取舍。

### 负面影响

- 需要自行维护更多接口、领域模型和编排代码。
- 不直接享受高层框架提供的全部集成组件。
- 流式取消、模型路由和 Trace 需要额外测试。
- pgvector 在超大规模或复杂多模态检索下可能需要迁移。

## 备选方案

### LangChain/LlamaIndex 全量编排

优点是开发初期快、现成组件多；缺点是关键语义更难控制和解释。P0 不采用。

### LangGraph

适合长时间、有状态、可恢复的 Agent。当前确定性 RAG Workflow 不需要其持久化执行能力，P0 不采用。

### Milvus

适合更大规模向量检索和专用运维场景。当前会增加容器、Schema 和一致性复杂度，P0 不采用。

### 云端 Embedding

效果和长文本能力可能更好，但增加额度、网络和复现依赖。P0 使用本地模型建立 Baseline，后续通过实验决定是否切换。

## 重新评估条件

出现以下情况时，应新增 ADR：

- 向量规模或延迟超过 pgvector 可接受范围。
- 需要真正的持久化 ReAct Agent。
- 本地 Embedding 在固定中英文评测上明显不足。
- MCP 工具需要写操作、人工审批或跨任务恢复。
- 项目需要多租户、复杂文档 ACL 或生产级高可用。
