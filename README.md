# Customer Agent 2

一个使用 Python 构建的、面向开源展示和求职简历的大模型应用工程项目。

项目目标不是再做一个“上传文档后调用模型”的演示，而是完整呈现从文档入库、问题理解、检索与重排序，到流式生成、引用溯源和效果评测的工程链路。

> 当前状态：M2-E 最小文档入库 API 已完成。项目已有 PostgreSQL/Redis 连接管理、
> 阿里云百炼 OpenAI-compatible Chat 非流式/流式适配器、本地
> `BAAI/bge-base-zh-v1.5` Embedding、版本化 pgvector 存储、Markdown/TXT 解析，以及
> 400/64 Token 分块、原子版本切换和同步上传/状态/删除 API；PDF/DOCX/CSV 和在线 RAG
> 仍是后续任务。

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
- OpenAI-compatible Async Client、httpx
- Sentence Transformers、PyTorch CPU
- pytest、pytest-asyncio

核心工作流不以 LangChain 或 LlamaIndex 为主骨架。详细原因参见 [ADR-0001](docs/adr/0001-technology-stack.md)。

## 默认模型策略

| 能力 | 默认方案 | 说明 |
|---|---|---|
| 最终回答 | `qwen3.7-max-preview` | 配置化，可随额度切换 |
| 内部快速任务 | `qwen3.7-flash` | 用于改写、意图、摘要等 |
| Embedding Baseline | `BAAI/bge-base-zh-v1.5` | 本地 CPU，768 维 |
| Rerank | `qwen3-rerank` | 未配置时允许显式降级 |
| VLM | `qwen3.7-plus` | 后续阶段启用 |

真实密钥只允许放在本地 `.env`，不得提交到仓库。

## 当前模型层边界

- Chat 同时定义非流式、流式、推理内容和 Token 用量结构。
- Chat 适配器使用异步连接池，支持独立首包超时，并在流结束、取消或调用方提前停止时释放响应。
- 供应商认证、额度、限流、超时、不可用和协议错误会转换为稳定且脱敏的领域错误。
- final 模型只用于最终回答，fast 模型供后续改写、意图和摘要等内部任务选择。
- Embedding 模型按首次请求懒加载，CPU 推理在工作线程执行，并串行保护同一个模型实例。
- Embedding 结果会验证批量形状、768 维、NaN/无限值和 L2 归一化；最大序列固定为 512 Token。
- Rerank 未启用时使用显式 No-op，保留原始顺序并记录降级原因，不使用 Chat 模型冒充 Rerank。
- Fake 模型可稳定复现正常结果、流式结果、排序结果和结构化错误，不需要网络或真实密钥。

Chat 协议目前通过本地 HTTP Mock 验证，没有调用真实云端模型或消耗额度。本地
Embedding 已使用模型缓存完成离线真实 Smoke，但模型权重不属于仓库内容。M2-A 至
M2-E 已把版本化存储、Markdown/TXT 解析、结构分块、批量 Embedding 和 pgvector 原子切换
连成同步 HTTP 入库闭环。项目还没有对外问答 API、PDF/DOCX/CSV 入库或在线向量检索。

## 当前文档解析边界

- 输入是尚未落盘的内存字节，默认单文件上限为可配置的 50 MiB。
- Markdown 支持 `.md`、`.markdown` 和对应 MIME；TXT 支持 `.txt` 和 `text/plain`。
- 当前只接受 UTF-8/UTF-8 BOM，拒绝空文件、超限文件、类型冲突、二进制控制字符和无有效文本。
- Markdown 保留标题层级、段落、列表项、代码块、章节路径和来源行号；TXT 保留段落和来源行号。
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

## 当前文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [系统架构](docs/ARCHITECTURE.md)
- [开发计划](docs/DEVELOPMENT_PLAN.md)
- [技术选型决策](docs/adr/0001-technology-stack.md)
- [文档版本与向量索引模式](docs/adr/0002-document-index-schema.md)
- [最小文档入库 API 契约](docs/adr/0003-minimal-ingestion-api.md)
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

Alembic 会确保 pgvector 扩展存在，并创建 M2-A 的 `knowledge_bases`、`documents`、
`document_versions` 和 `chunks` 表。迁移只建立存储模式，不会自动解析或导入文档。

```powershell
alembic upgrade head
```

### 5. 启动 API

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

上传是同步操作，本地 CPU Embedding 完成前请求会保持。当前只接受 UTF-8 Markdown/TXT，
默认上限 50 MiB。

### 7. 运行质量检查

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

如需用本地已迁移的 PostgreSQL 验证真实向量写入、Cosine 查询和版本唯一约束：

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

## 参考与致谢

本项目参考了 [Ragent](https://github.com/nageoffer/ragent) 在文档入库、问题理解、多路检索、模型路由、MCP 与流式回答方面的设计。Customer Agent 2 使用 Python 重新组织实现，并通过独立测试与评测验证行为。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。第三方项目归属说明参见 [NOTICE](NOTICE)。
