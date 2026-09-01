# Customer Agent 2

一个使用 Python 构建的、面向开源展示和求职简历的大模型应用工程项目。

项目目标不是再做一个“上传文档后调用模型”的演示，而是完整呈现从文档入库、问题理解、检索与重排序，到流式生成、引用溯源和效果评测的工程链路。

> 当前状态：M5-C 已固定原项目附带的 150 条评测集和 116 篇检索语料，并完成 132 条本地
> Retrieval OFF、132/132 次真实 `qwen3-rerank` ON 单变量实验，以及 150/150 次真实
> `qwen3.8-flash` Intent 评测。项目已有
> PostgreSQL/Redis 连接管理、
> 阿里云百炼 OpenAI-compatible Chat 非流式/流式适配器、本地
> `BAAI/bge-base-zh-v1.5` Embedding、版本化 pgvector 存储、Markdown/TXT/PDF/DOCX/CSV
> 解析，以及 400/64 Token 分块、原子版本切换、同步上传/状态/删除 API 和带作用域过滤的
> active-only Cosine 向量召回。`POST /api/v1/chat/stream` 已将
> `保存用户消息 → 加载摘要与最近 6 轮 → 改写/拆分 → Intent 路由 → 检索 → 加权 RRF/去重 → 可配置 Rerank/TopK → 安全 Prompt → Chat 流 → 保存回答/Trace`
> 通过版本化 SSE 契约公开；超过 12 个 completed 轮次后会用 fast 模型增量摘要滑出窗口的旧轮次。
> M5-D 已离线完成 Intent 失败切片，并把真实验证改为带缓存的 22 条失败集 → 追加 18 条回归集
> → 追加 110 条完整集。v2 第一阶段因 1 条 under-retrieval 未通过门禁并停止；v3 前 40 条全部
> 正确，第三阶段只新增剩余 110 次真实调用后累计达到 149/150、0 调用失败，但完整集出现 1 条
> under-retrieval，未通过预登记的安全门禁，因此候选不晋级，线上默认继续使用 `m4-c-v1`。v4
> 已完成离线设计，把后续付费验证缩小为严格串行的 4 条已知边界探针和 20 条全新冻结挑战样本，
> 第一阶段 4/4 真实调用成功且全部正确，三类错误均为 0；当前停在独立的 20 条挑战集费用边界。

## 项目目标

- 使用显式 Python Pipeline 实现可调试的 RAG 主流程。
- 支持本地 Embedding 与云端大模型的组合部署。
- 通过固定评测集验证 Intent、Retrieval 和 Rerank，而不是只展示主观问答效果。
- 保留清晰的接口边界，便于后续扩展 MCP、VLM、关键词检索和知识图谱。
- 形成能够在面试中解释设计、失败场景、替代方案和实验结果的开源项目。

## 规划中的核心链路

```text
文档上传
  → 类型识别与解析
  → 结构化分块
  → Embedding
  → PostgreSQL/pgvector 索引

用户问题
  → 会话记忆
  → Query Rewrite / 问题拆分
  → 意图识别 / 歧义澄清
  → 检索与候选融合
  → Rerank / TopK
  → Prompt 与引用组装
  → LLM 流式回答
```

## 技术选型

- Python 3.11
- FastAPI、Pydantic、Uvicorn
- SQLAlchemy 2、Alembic、asyncpg
- PostgreSQL、pgvector
- Redis
- markdown-it-py
- python-multipart
- pypdf、python-docx
- OpenAI-compatible Async Client、httpx
- Sentence Transformers、PyTorch CPU
- pytest、pytest-asyncio

核心工作流不以 LangChain 或 LlamaIndex 为主骨架。详细原因参见 [ADR-0001](docs/adr/0001-technology-stack.md)。

## 默认模型策略

| 能力 | 默认方案 | 说明 |
|---|---|---|
| 最终回答 | `qwen3.7-max-preview` | 配置化，可随额度切换 |
| 内部快速任务 | `qwen3.8-flash` | 用于改写、意图、摘要等；结构化任务关闭思考模式 |
| Embedding Baseline | `BAAI/bge-base-zh-v1.5` | 本地 CPU，768 维 |
| Rerank | `qwen3-rerank`，默认关闭 | 关闭或已知失败时显式降级 |
| VLM | `qwen3.7-plus` | 后续阶段启用 |

真实密钥只允许放在本地 `.env`，不得提交到仓库。

## 当前模型层边界

- Chat 同时定义非流式、流式、推理内容和 Token 用量结构。
- Chat 适配器使用异步连接池，支持独立首包超时，并在流结束、取消或调用方提前停止时释放响应。
- 供应商认证、访问权限、额度、限流、超时、不可用和协议错误会转换为稳定且脱敏的领域错误；
  403 中的已知额度代码优先归为额度错误，不会被笼统归为权限错误。
- final 模型只用于最终回答；fast 模型已用于长会话摘要、严格 JSON Query Rewrite 和 Intent 分类。
- Embedding 模型按首次请求懒加载，CPU 推理在工作线程执行，并串行保护同一个模型实例。
- Embedding 结果会验证批量形状、768 维、NaN/无限值和 L2 归一化；最大序列固定为 512 Token。
- Rerank 未启用时使用显式 No-op；显式启用后使用异步 `qwen3-rerank` 专用接口和独立连接池，
  不使用 Chat 模型冒充 Rerank。
- 配置 Workspace ID 时，Chat 与 Rerank 自动使用同一地域的业务空间专属域名；没有 Workspace
  ID 时，Chat 才回退到 `DASHSCOPE_BASE_URL`。这避免 Workspace Key 被误发到公共 Chat 端点。
- Fake 模型可稳定复现正常结果、流式结果、排序结果和结构化错误，不需要网络或真实密钥。

Chat 协议已通过本地 HTTP Mock 覆盖，并由 150 条真实 `qwen3.8-flash` Intent 运行验证非流式
云端路径。本地 Embedding 已使用模型缓存完成离线真实 Smoke，但模型权重不属于仓库内容。M2-A 至
M2-G 已把五种 P0 文档解析、版本化存储、结构分块、批量 Embedding、pgvector 原子切换和
在线向量召回连成闭环。M3-A 已接入最终 Chat 流，M3-B 已提供公开 SSE 问答 API，M3-C 已保存
最小会话消息和 RAG Run 终局；M4-A 已把持久化摘要和最近 completed 消息接入 Prompt，M4-B
已让上下文改写和最多 3 个子问题实际参与向量召回；M4-C 已实现系统直答、知识库问答、需要
澄清三类路由和固定 20 条决策 Smoke Test；M5-A 已把加权 RRF、重复控制和候选预算接入在线
Pipeline。M5-B 已增加真实 Rerank 适配器、Workspace 配置、资源释放和固定 20 条 OFF/ON
运行器；最终一轮 20/20 真实请求成功且无重试。固定合成集的 OFF → ON 指标为：Hit@1
`0.10 → 1.00`、Hit@3 `0.30 → 1.00`、MRR@10 `0.2929 → 1.00`，总计 9,686 Token，成功请求
延迟 P50/P95 为 196.685/231.871 ms。该结果只验证工程链路与合成小样本方向，不代表生产效果。

M5-C 使用原项目附带数据的字节级快照重新建立了独立 Baseline：150 条 Query 中 132 条进入
Retrieval 分母、18 条 no-rag 不混入检索指标；115 篇正式文档加 `PRODUCT_MAPPING` 共 116 篇，
已通过正式解析、400/64 分块、本地 BGE 和 pgvector 幂等导入为 1,524 个 active Chunk。132 条
真实本地 OFF 运行 0 失败，Doc Hit@1/3/5/10 分别为 `0.7045/0.8939/0.9545/0.9848`，
Recall@3/5/10 为 `0.6723/0.7683/0.8447`，MRR@10 为 `0.8075`。42 条没有召回全部 required
文档，其中 2 条 Top10 完全未命中。随后 132/132 次真实 `qwen3-rerank` 请求成功、0 失败、
零重试，共 210,769 Token，延迟 P50/P95 为 199.335/266.688 ms；ON 的
Hit@1/3/5/10 为 `0.6364/0.8712/0.9470/0.9773`，Recall@3/5/10 为
`0.6578/0.7689/0.8327`，MRR@10 为 `0.7622`，胜/平/负为 21/81/30。该固定配置下 Rerank
没有提升完整集，且 Top10 文档多样性略降；因此默认仍保持关闭。脱敏逐样本报告见
`evaluation/reports/m5c-full-retrieval-off.json` 和
`evaluation/reports/m5c-full-retrieval-off-on.json`。

完整 Intent 最终独立运行 150/150 成功、0 降级、零重试，固定模型为 `qwen3.8-flash`，温度 0、
关闭思考模式、阈值 0.75/0.10、最大输出 256 Token；评测专用单条超时为 60 秒，线上默认仍为
20 秒。总体准确率为 `128/150 = 85.33%`，132 条 RAG 为 `121/132 = 91.67%`，18 条 no-rag 为
`7/18 = 38.89%`；SUPPORT/FEEDBACK/CHAT 分层准确率为 `92.80%/33.33%/70.00%`。错误包括
11 条 RAG 被送去澄清，以及 no-rag 的 5 条误进知识库、6 条转澄清。输入/输出合计
45,101/7,366 Token，延迟 P50/P95 为 794.040/2,459.129 ms。该结果表明 RAG 路由已较稳定，
但 no-rag 和 FEEDBACK 仍是下一轮阈值/意图树校准重点。脱敏报告见
`evaluation/reports/m5c-full-intent.json`。

M5-D 离线分析确认 22 条错误中有 17 条进入不必要的澄清、5 条 no-rag 被送入知识库，并且没有
应检索却直接回答的错误。分析只保存聚合计数和 Query ID，见
`evaluation/reports/m5d-intent-failure-analysis.json`。评测专用
`evaluation/config/m5d-intent-tree-v2.json` 第一阶段 22/22 成功、0 失败、零重试，应检索/no-rag
各正确 9/11，总计 18/22；但出现 1 条 under-retrieval，违反安全门禁，因此没有执行回归集。
脱敏结果见 `evaluation/reports/m5d-intent-v2-failures.json`。v3 只根据这四条错误进一步区分选购、
库存、隐含功能建议和竞品问题，文件为 `evaluation/config/m5d-intent-tree-v3.json`。v3 第一阶段
22/22 成功、0 失败、零重试，应检索/no-rag 均为 11/11，under-retrieval、over-retrieval 和
clarification 均为 0；脱敏结果见 `evaluation/reports/m5d-intent-v3-failures.json`。该失败集已用于
开发调优，单独的 22/22 不能证明泛化提升。第二阶段新增 18 条 M5-C 原本正确的保护样本也达到
18/18，累计 40/40，SUPPORT/FEEDBACK/CHAT 为 15/15、15/15、10/10，仍为 0 under-retrieval、
0 over-retrieval、0 clarification；脱敏累计报告见
`evaluation/reports/m5d-intent-v3-guard.json`。线上默认仍是 `m4-c-v1`，只有完整 150 条通过才会
讨论晋级。第三阶段复用前 40 条缓存，只新增 110 次真实调用，最终 150/150 请求成功、零自动重试，
总体、RAG、no-rag 正确数分别为 149/150、131/132、18/18，SUPPORT/FEEDBACK/CHAT 为
124/125、15/15、10/10；输入/输出共 84,851/6,747 Token，延迟 P50/P95 为
693.270/961.267 ms。唯一错误 `S3-08` 应进入知识库，却以 0.95 高置信进入系统直答，形成 1 条
under-retrieval；其余 over-retrieval 和 clarification 均为 0。尽管准确率明显高于 M5-C，安全
门禁要求 under-retrieval 必须为 0，因此 v3 不晋级且不会影响线上请求。脱敏阶段报告和标准完整
报告分别见 `evaluation/reports/m5d-intent-v3-full.json`、
`evaluation/reports/m5d-full-intent-v3.json`。

v4 离线分析确认 `S3-08` 属于“自营具体商品与第三方竞品混合对比”，不能与“纯第三方问题或泛化
品牌主观评价”共用直接回答边界。`evaluation/config/m5d-intent-tree-v4.json` 只细化这条通用语义，
不改模型、阈值、选择算法或线上树。验证清单
`evaluation/config/m5d-intent-v4-validation.json` 先使用 4 条已查看的相邻边界作开发探针，再使用
20 条未原样复用原 150 条问题的冻结挑战样本；挑战集保持 knowledge-base/system-direct 10/10。
两个阶段都要求全部正确且 under-retrieval、over-retrieval、clarification 均为 0。该挑战集是人工
构造的边界覆盖，不是生产流量的无偏随机样本；即使通过也只支持进入影子或小流量验证，不会自动
替换线上 `m4-c-v1`。第一阶段已完成 4/4、0 失败、零自动重试，检索/直答均为 2/2，三类错误均
为 0；记录 2,589 输入 Token、182 输出 Token，延迟 P50/P95 为 672.969/1,020.245 ms。脱敏报告
见 `evaluation/reports/m5d-intent-v4-development.json`，协议见 ADR-0014。

## 当前文档解析边界

- 输入是尚未落盘的内存字节，默认单文件上限为可配置的 50 MiB。
- Markdown 支持 `.md`、`.markdown`，TXT 支持 `.txt`，CSV 支持 `.csv`；三者只接受
  UTF-8/UTF-8 BOM。
- PDF 和 DOCX 同时校验扩展名/MIME 与二进制签名，拒绝伪造文件、加密 PDF、宏 DOCX 和
  高风险压缩包。
- 所有格式都拒绝空文件、超限文件、类型冲突和无有效文本。
- Markdown 保留标题层级、段落、列表项、代码块、章节路径和来源行号；TXT 保留段落和来源行号。
- PDF 按页保留来源，DOCX 保留标题、段落、列表和表格记录，CSV 为每条记录重复表头。
- 扫描 PDF/OCR、旧版 DOC、XLS/XLSX、图片和自动编码/分隔符猜测尚未支持。
- 解析器通过领域接口注册和选择，不依赖 FastAPI、数据库或文件系统。

这些能力已连接到同步上传 API。应用不持久化原文件；multipart 实现可能在请求期间使用
框架管理的临时 spool，请求结束时显式关闭。

## 当前分块边界

- 使用与默认 Embedding 相同且固定 revision 的 BGE Tokenizer，不用字符数估算 Token。
- 已确认 Baseline 为目标 400 Token、超预算滑窗 Overlap 64 Token，且 400 不超过模型 512 Token 上限。
- 优先按标题章节、段落、列表项和代码块组合；不同标题章节不会合并。
- 只有结构块或结构组超过 400 Token 时才执行 Token 滑窗；自然结构边界不强制制造重复文本。
- Chunk 草稿包含连续索引、Token 数、内容哈希、章节路径、Block 范围、来源行号和实际 Overlap。

分块输出可由 M2-D 入库用例批量调用 Embedding 并写入 pgvector。

## 当前入库边界

- 输入仍是内存中的 `DocumentSource`，由调用方提供知识库 ID 和稳定 `source_key`。
- Parse 和 Chunk 完成后才创建 `building` 版本；Embedding 期间不占用长数据库事务。
- 分块以一个批量请求交给本地 Embedding 适配器，适配器按配置的 CPU Batch 执行。
- 激活事务同时写入全部 Chunk、将旧 active 版本改为 `superseded`、将新版本改为 `active`。
- Embedding 失败、协议异常和取消会使新版本进入 `failed`；旧 active 版本不受影响。
- 当前保留 `failed` 和 `superseded` 版本用于追溯，清理策略、任务状态 API 和对象存储仍属后续任务。

## 当前入库 API

- `POST /api/v1/knowledge-bases`：创建固定使用当前 Embedding Baseline 的知识库。
- `POST /api/v1/knowledge-bases/{knowledge_base_id}/documents`：multipart 上传 `file`，可选
  `source_key`；只有新版本已经 active 才返回 HTTP 201。
- `GET /api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}`：返回最新入库尝试和
  当前 active 版本。
- `DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}`：级联硬删除文档、版本和 Chunk。

当前 API 是受控开发/演示边界：没有身份认证、多租户、后台任务、进度轮询和对象存储。删除不可恢复。

## 当前向量检索边界

- `VectorRetrievalService` 只依赖类型化 Embedding 和检索仓储接口；SQLAlchemy、pgvector 与
  JSONB 细节留在基础设施层。
- 每次请求必须显式提供非空知识库 ID 列表；文档 ID、文档格式、解析器、章节和页码过滤均在
  PostgreSQL 查询侧完成。未来“全局检索”也必须先由权限层解析为明确的可访问知识库列表。
- 查询向量的 model ID、revision、维度和归一化必须与范围内每个知识库完全一致，不一致会返回
  稳定错误，不会静默跳过。
- 只检索 `active` 文档版本；`building`、`failed` 和 `superseded` Chunk 永远不会进入候选。
- Cosine HNSW 查询默认召回预算为 20、`ef_search` 为 100；带过滤查询在单次事务内启用
  `strict_order` 迭代扫描，设置不会泄漏到连接池中的后续请求。

该能力已经接入在线 Pipeline，并由 SSE API 在请求提供的显式作用域内调用。最多 3 个子问题
并发执行同作用域检索；M5-A 以等权加权 RRF（`k=60`）按 Chunk UUID 聚合证据，再按内容哈希
去重、限制每个文档最多 2 个 Chunk，并把最多 40 个候选交给 Rerank 边界，最终保留 TopK 10。

## 当前 RAG Pipeline、记忆与 SSE API 边界

- 使用请求级 `ChatPipelineContext` 显式传递原问题、当前改写结果、检索结果、TopK、Prompt、
  引用来源和阶段 Trace，不使用全局可变状态。
- 当前加载可选持久化摘要和最近 6 个 completed/clarification user/assistant 完整轮次；running、
  no_context、failed、cancelled 和残缺消息不会进入 Prompt。历史只用于理解指代，不作为知识事实来源。
- 累计完整对话不超过 12 轮时不调用摘要模型；超过后，每个滑出最近 6 轮窗口的完整轮次
  使用 fast Chat 增量合并进摘要。摘要失败保留旧摘要并降级到最近消息，不推翻成功回答。
- fast 模型把当前问题改写为可独立理解的问题，并输出 1～3 个不重复检索子问题；输出必须是
  严格 JSON。默认改写超时 20 秒、最多 512 token，模型错误、超时或协议不合规时退回原问题，
  并在日志与 Trace 保留稳定降级代码。
- 子问题使用结构化并发检索，异常、超时和取消会收拢全部子任务。检索结果使用等权加权 RRF，
  再执行内容哈希去重、每文档上限和候选截断；默认 No-op Rerank 保留融合顺序，显式启用后调用
  `qwen3-rerank`，两条路径最终都返回 TopK 10。
- fast 模型按打包的三节点意图树输出三个独立置信分数。最高分至少 `0.75` 且与第二名差值至少
  `0.10` 才执行系统直答或知识库问答；低置信、歧义或显式澄清路由会返回 `guidance`，不检索也
  不调用 final 模型。分类模型失败、超时或协议不合规时，只在请求已授权的知识库作用域内降级检索。
- 系统直答跳过检索且不产生 `sources`；其 Prompt 明确禁止声称访问过知识库、订单、工具或网络。
- Prompt 把文档标记为不可信资料，转义文档内容和来源属性，要求回答使用 `[1]` 形式引用；
  模型的 reasoning 增量不会作为答案事件向上游暴露。
- 空检索以 `no_context` 正常结局短路，不调用 Chat 模型，因此不会在没有资料时生成答案。
- Pipeline 使用可配置的 `RAG_GLOBAL_TIMEOUT_SECONDS` 单一截止时间约束改写、检索、融合、Rerank
  和模型流；Rerank 另有默认 10 秒独立上限。模型流协议要求可关闭的异步生成器，超时、取消或
  调用方提前停止时会逐层执行 `aclose()`。
- `POST /api/v1/chat/stream` 要求非空知识库 ID 列表，返回
  reply_to/status/content/sources/guidance/error/done 事件、严格递增序号和 `X-Request-ID`。首次请求省略
  `conversation_id`，后续请求用 reply_to 返回的 ID 继续会话。
- 每个已开始请求会保存 user 消息与 RAG Run；completed 会原子保存 assistant 消息、可选来源 Chunk、
  模型用量和轻量 Trace；clarification 会保存 guidance assistant 消息；no_context/failed/cancelled
  不保存伪成功回答。Run 同时保存稳定 `intent_route`。
- 当前没有应用用户认证、会话管理 API、断线重放或心跳；真实云端 Rerank 已完成工程接线和
  固定 20 条 Smoke。协议、Run 持久化、记忆、改写、Intent 和检索后处理边界分别见
  ADR-0005 至 ADR-0011。

## 当前文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [系统架构](docs/ARCHITECTURE.md)
- [开发计划](docs/DEVELOPMENT_PLAN.md)
- [技术选型决策](docs/adr/0001-technology-stack.md)
- [文档版本与向量索引模式](docs/adr/0002-document-index-schema.md)
- [最小文档入库 API 契约](docs/adr/0003-minimal-ingestion-api.md)
- [多格式文档安全解析边界](docs/adr/0004-multiformat-document-parsing.md)
- [流式 RAG API 与 SSE 事件契约](docs/adr/0005-streaming-rag-api.md)
- [最小会话消息与 RAG Run 持久化](docs/adr/0006-conversation-rag-run-persistence.md)
- [会话记忆与 M4 意图基线参数](docs/adr/0007-conversation-memory-baseline.md)
- [Query Rewrite 与多问题检索基线](docs/adr/0008-query-rewrite-and-multi-question-retrieval.md)
- [Intent 路由、澄清与降级契约](docs/adr/0009-intent-routing-and-guidance.md)
- [检索后处理与 No-op Rerank 基线](docs/adr/0010-retrieval-post-processing-baseline.md)
- [DashScope Rerank 与 20 条 OFF/ON Smoke](docs/adr/0011-dashscope-rerank-and-smoke-evaluation.md)
- [完整评测数据与协议](docs/adr/0012-full-evaluation-dataset-and-protocol.md)
- [Intent 失败切片与候选树实验](docs/adr/0013-intent-failure-calibration.md)
- [v4 竞品边界聚焦验证](docs/adr/0014-intent-v4-focused-validation.md)
- [AI 协作规则](AGENTS.md)

## 本地启动

### 1. 安装项目

项目使用 Python 3.11。推荐使用锁文件和项目配置的 PyTorch CPU-only 索引：

```powershell
uv sync --locked --all-extras
```

当前开发机也可以继续使用已经包含 CPU 版 PyTorch 的 `customer` Conda 环境：

```powershell
conda activate customer
python -m pip install "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

### 2. 创建本地配置

```powershell
Copy-Item .env.example .env
```

`.env` 只用于本地，已经被 Git 忽略。请在其中填写真实模型密钥，不要修改
`.env.example` 保存个人凭据。

### 3. 启动 PostgreSQL/pgvector

```powershell
docker compose up -d postgres
docker compose ps
```

当前开发机已有 Redis 监听 `127.0.0.1:6379`，因此本项目 Compose 不重复创建。
如果本机 Redis 地址不同，请只在本地 `.env` 中修改 `REDIS_URL`。

### 4. 执行数据库迁移

Alembic 会确保 pgvector 扩展存在，创建 M2 的 `knowledge_bases`、`documents`、
`document_versions`、`chunks`，M3-C 的 `conversations`、`messages`、`rag_runs`，以及 M4-A 的
`conversation_summaries`。M4-C 为 `rag_runs` 增加 `intent_route`，并增加独立 `clarification` 终局。
迁移只建立存储模式，不会自动解析或导入文档。

```powershell
alembic upgrade head
```

### 5. 启动 API

默认服务图会在启动时创建最终 Chat 模型连接池，因此需要先在本地 `.env` 填写
`DASHSCOPE_API_KEY`。不要把真实密钥写入 `.env.example` 或提交到 Git。

```powershell
python -m uvicorn customer_agent2.main:app --reload
```

验证健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

验证就绪检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/health` 只说明 API 进程存活，不访问外部服务；`/ready` 只有在 PostgreSQL、
pgvector 和 Redis 都可用时返回 HTTP 200，否则返回不包含连接串和底层异常的 HTTP 503。

OpenAPI 页面位于 `http://127.0.0.1:8000/api/v1/docs`。

### 6. 调用最小入库 API

```powershell
$knowledgeBase = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/knowledge-bases `
  -ContentType "application/json" `
  -Body '{"slug":"demo-docs","name":"Demo Docs"}'

$upload = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/knowledge-bases/$($knowledgeBase.id)/documents" `
  -Form @{file = Get-Item .\guide.md; source_key = "manual/guide.md"}

Invoke-RestMethod `
  "http://127.0.0.1:8000/api/v1/knowledge-bases/$($knowledgeBase.id)/documents/$($upload.document_id)"
```

上传是同步操作，本地 CPU Embedding 完成前请求会保持。当前接受 Markdown、TXT、PDF、
DOCX 和 UTF-8 CSV，默认单文件上限 50 MiB；详细解析上限见 `.env.example`。

### 7. 调用流式问答 API

```powershell
$body = @{
  question = "如何申请退款?"
  scope = @{knowledge_base_ids = @($knowledgeBase.id)}
} | ConvertTo-Json -Depth 4

curl.exe -N `
  -H "Content-Type: application/json" `
  -d $body `
  http://127.0.0.1:8000/api/v1/chat/stream
```

HTTP 200 表示 SSE 流已经建立；最终结果以 `done` 事件的 `outcome` 为准。来源事件只返回引用
坐标和内容哈希，不返回完整文档正文。首次返回的 `reply_to.conversation_id` 可放入下一次请求
顶层的可选 `conversation_id`；当前同一会话不允许并发运行两个请求。

### 8. 运行质量检查

```powershell
ruff check .
ruff format --check .
pyright
pytest
uv lock --check
uv pip check
uv audit
```

如果本机已经缓存默认模型，可执行不会联网下载的真实 CPU Smoke：

```powershell
$env:RUN_LOCAL_MODEL_SMOKE="1"
pytest -m model_smoke
```

如需用本地已迁移的 PostgreSQL 和 Redis 验证真实向量写入、过滤检索、Cosine 排名、
active-only 版本语义、基础 RAG Pipeline、会话/RAG Run 持久化、最近记忆/摘要窗口、Query
Rewrite、Intent 路由/澄清和 Fake Chat 驱动的公开 SSE 端到端路径：

```powershell
$env:RUN_DATABASE_INTEGRATION="1"
pytest -m database_integration
```

如果同时具备已缓存的默认模型和已迁移的 PostgreSQL，可验证真实
Markdown 解析、BGE 分块/Embedding 和 pgvector 原子激活：

```powershell
$env:RUN_INGESTION_SMOKE="1"
pytest -m ingestion_smoke
```

M5-B 的 Rerank Smoke 使用 20 条固定合成样本。先在本地 `.env` 配置同一地域/计费方案下有效的
`DASHSCOPE_API_KEY`、`DASHSCOPE_WORKSPACE_ID` 和 `DASHSCOPE_RERANK_REGION`，再显式授权真实
调用。配置 Workspace ID 后，Chat 自动使用
`https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1`，Rerank 使用同一域名下的
`compatible-api/v1`；不需要把真实 Workspace ID 写进 `DASHSCOPE_BASE_URL`：

```powershell
python -m customer_agent2.evaluation --live
```

OFF 不访问云端；ON 串行发出最多 20 次 `qwen3-rerank` 请求，每条最多一次且不重试。报告只包含
固定样本 ID、名次、聚合指标、延迟、Token 和稳定错误码，不包含密钥、Workspace ID、Query 或
候选正文。认证、配置、额度或协议错误会在首条失败后终止整轮。

M5-C 的固定快照位于 `evaluation/datasets/ragenteval-v1/`，来源和 20 条遗留快速集与全量记录的
差异边界见目录内 `SOURCE.md`。150 条全量文件是唯一评测真值。先执行完全离线、可幂等复跑的
语料导入：

```powershell
python -m customer_agent2.evaluation.full_evaluation_cli --prepare-corpus --offline-models
```

首次运行导入 116 篇；相同快照再次运行应跳过 116 篇且不创建重复版本。随后执行不访问云端的
132 条 Retrieval OFF：

```powershell
python -m customer_agent2.evaluation.full_evaluation_cli --run-off --offline-models
```

固定参数为 400/64 Token 分块、向量召回 20、HNSW `ef_search=100`、RRF `k=60`、每文档最多
2 个 Chunk、候选上限 40、最终 TopK 10。400/64 在 BGE 512 Token 上限内保留跨块上下文；召回
20 与 `ef_search=100` 保持当前正式检索成本；RRF 60 降低单个原始名次的过强影响；每文档 2 个
Chunk 防止单文档挤占上下文；40→10 给 Rerank 足够候选并限制请求体和最终 Prompt。

真实 OFF/ON 必须显式确认最大付费调用数，且会复用每条 Query 的同一次向量召回和融合候选：

```powershell
python -m customer_agent2.evaluation.full_evaluation_cli `
  --live-rerank --accept-paid-calls 132 --offline-models
```

没有 `--live-rerank` 和精确的 `--accept-paid-calls 132` 时，运行器不会调用云端。ON 串行、每条
最多一次、不重试，认证、配置、额度、协议错误或任何降级会在首条发生时安全终止。报告不保存
API Key、Workspace ID、Query、Ground Truth 或 Chunk 正文。协议详见
[ADR-0012](docs/adr/0012-full-evaluation-dataset-and-protocol.md)。

已完成的真实 OFF/ON 报告严格按最终 TopK 的 Chunk 排名计算 Doc 指标：先截取前 10 个 Chunk，
再按首次出现顺序做文档去重。付费 ON 完成后发现旧 OFF 辅助统计曾继续扫描 Top10 之外的 Chunk，
因此 ON 结果原样保留，OFF 使用相同固定配置做了确定性本地复算；没有为修正统计追加云端调用。

完整 Intent 使用全部 150 条，`requires_rag=true` 映射 `knowledge_base`，18 条 no-rag 映射
`system_direct`；原数据没有可靠的 `clarification` 真值，因此不会伪造该路由的准确率。真实运行
同样需要独立确认最多 150 次 fast Chat 调用：

```powershell
python -m customer_agent2.evaluation.full_intent_cli `
  --live-intent --accept-paid-calls 150 --intent-timeout-seconds 60
```

运行器记录总体、RAG/no-rag、Intent L1 切片、混淆矩阵、P50/P95 和 Token；不保存问题或答案，
任何模型降级在第一条发生时终止。每条成功结果会先原子写入脱敏 checkpoint；再次执行时只允许
同一数据集、模型和参数的连续成功前缀，并要求 `--accept-paid-calls` 精确等于剩余条数，避免重复
产生费用。60 秒只覆盖本次评测，线上配置仍使用默认 20 秒。该命令与 132 次 Rerank 是两个独立
费用边界，不能互相代替授权。

M5-D 的失败切片完全离线运行：

```powershell
python -m customer_agent2.evaluation.intent_failure_analysis_cli
```

候选树真实实验使用独立、严格绑定候选配置的缓存，避免覆盖 M5-C 基线。v2 已因第一阶段门禁失败
停止；v3 已完成 22 条失败集、新增 18 条回归保护和剩余 110 条，缓存中的 150 条不应重复运行。
第三阶段实际使用的命令为：

```powershell
python -m customer_agent2.evaluation.staged_intent_cli `
  --live-intent --stage full --accept-paid-calls 110 `
  --intent-timeout-seconds 60
```

该次运行已经完成；除非建立新候选、独立缓存和新的费用授权，否则不要再次执行这条付费命令。

每次授权数必须等于缓存中当前缺少的条数；例如第一阶段已成功 5 条后中断，续跑必须确认 17，不能
仍写 22 或 150。每条成功结果先原子落盘，同一候选的成功样本不会跨阶段重复调用；失败尝试可能
产生费用但不会伪装成成功，续跑前会重新显示准确缺口。修改模型、候选树、阈值、超时、阶段清单
或基线后旧缓存会被拒绝，应为新候选指定新的 `--cache`、`--output` 和
`--full-output` 路径。阶段报告保存脱敏三路分数，使一次付费运行可以离线重放其他阈值；同一完整
集上的阈值重放只算探索性分析，不能替代新的留出集验证。详细控制变量和晋级门槛见 ADR-0013。
运行期间同一缓存有独占锁，避免两个终端对相同缺口重复付费；若进程被强制结束并遗留 `.lock`
文件，应先确认没有同一候选评测仍在运行，再手动删除锁文件。阶段报告中的尝试数来自本地逐条
回调，用于开发审计但不能替代供应商账单；强制结束进程等极端窗口仍应以控制台账单为准。

v4 不再重跑完整 150 条。第一阶段已经按以下命令完成 4 条开发探针：

```powershell
python -m customer_agent2.evaluation.candidate_intent_validation_cli `
  --live-intent --stage development --accept-paid-calls 4 `
  --intent-timeout-seconds 60
```

第一阶段已达到 4/4 且三类错误计数均为 0，运行器现已解锁但不会自动执行 20 条冻结挑战集。只有
得到新的独立费用授权后，才使用：

```powershell
python -m customer_agent2.evaluation.candidate_intent_validation_cli `
  --live-intent --stage challenge --accept-paid-calls 20 `
  --intent-timeout-seconds 60
```

两个阶段使用同一份绑定模型、树语义哈希、清单内容和超时参数的本地缓存；成功样本不会重复调用，
授权数必须精确等于当前阶段缺口。报告不保存问题或模型原文。当前第一阶段缓存为 4 条，第二条
命令尚未执行；完整限制和后续影子验证边界见 ADR-0014。

## 参考与致谢

本项目参考了 [Ragent](https://github.com/nageoffer/ragent) 在文档入库、问题理解、多路检索、模型路由、MCP 与流式回答方面的设计。Customer Agent 2 使用 Python 重新组织实现，并通过独立测试与评测验证行为。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。第三方项目归属说明参见 [NOTICE](NOTICE)。
