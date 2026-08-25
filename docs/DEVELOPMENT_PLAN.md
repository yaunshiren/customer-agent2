# Customer Agent 2 开发计划

## 1. 执行原则

- 先建立可运行的最小纵向链路，再增加高级能力。
- 每个阶段都有可验证产物和停止条件。
- 不以“代码已经写完”作为完成标准，必须运行测试或评测。
- 一次只改变一个检索或模型变量，避免无法解释效果变化。
- 没有通过当前阶段验收前，不同时扩展多个 P1/P2 能力。

## 2. 总体里程碑

| 阶段 | 目标 | 预计有效开发日 |
|---|---|---:|
| M0 | 文档、规则和工程脚手架 | 1～2 |
| M1 | 模型、配置与基础设施 | 2～3 |
| M2 | 文档入库与向量索引 | 2～4 |
| M3 | 基础流式 RAG | 2～3 |
| M4 | Rewrite、Intent、Memory 与引用 | 2～3 |
| M5 | RRF、Rerank、评测与加固 | 3～4 |
| P1 | MCP、VLM、关键词与模型容错 | 后续迭代 |

P0 推荐版本总计约 12～18 个有效开发日。外部模型额度、Docker 下载和环境权限等待不计入纯开发时间。

当前已完成 M2-A 和 M2-B：数据库已有版本化文档与 768 维向量模式；应用已有类型化
文档契约、Markdown/TXT 安全识别、解析器选择，以及保留章节和来源行号的结构化输出。
PDF/DOCX/CSV、分块、入库编排与 API 尚未开始。

## 3. M0：项目地基

### 任务

- 完成 AGENTS、README、Scope、Architecture、Development Plan 和 ADR。
- 创建 Python 3.11 项目结构和 `pyproject.toml`。
- 配置 Ruff、Pyright、pytest 和基础 CI。
- 创建安全的配置模型和 `.env.example`。
- 创建 Docker Compose：PostgreSQL/pgvector，按需增加 MinIO。
- 建立健康检查和测试目录。

### 验收

- 全新虚拟环境可以安装依赖。
- 应用可以启动并返回健康检查。
- 配置缺失时产生清晰错误。
- CI 可以执行格式检查、类型检查和空测试套件。

### 停止条件

脚手架、配置、数据库容器或测试命令任一不可复现时，不进入 M1。

## 4. M1：模型与基础设施

### 任务

- 实现统一 Chat、Embedding 和 Rerank 接口。
- 接入阿里云 OpenAI-compatible Chat 流式接口。
- 区分 final/fast 模型配置。
- 接入本地 `BAAI/bge-base-zh-v1.5`。
- 验证 768 维、最大 512 Token、归一化和 CPU Batch。
- 实现 Rerank No-op 适配器，为云端 Rerank 预留接口。
- 建立 SQLAlchemy、Alembic 和 Redis 连接管理。
- 增加外部 HTTP Mock 和模型 Smoke Test。

### 验收

- Chat 可以返回非流式和流式结果。
- Embedding 批量输出形状、维度和数值合法。
- 模型超时、无额度和协议错误可以区分。
- 连接池在应用关闭时正确释放。

### 风险检查

- `qwen3.7-max-preview` 仅思考模式，内部任务应使用快速模型。
- 本地模型偏中文，英文长文档效果需要后续实验验证。

## 5. M2：文档入库

### 任务

- 建立知识库、文档、版本和 Chunk 数据模型。
- 实现 Markdown、TXT、PDF、DOCX、CSV 解析器。
- 实现结构感知分块与 Token 超限二次切分。
- 保存来源元数据和内容哈希。
- 批量 Embedding 并写入 pgvector。
- 实现同一文档重建、替换和失败回滚。
- 建立最小上传、状态查询和删除 API。

### 验收

- 选择每种格式至少一个固定样本完成入库。
- 重复上传不会产生重复有效 Chunk。
- 解析失败不留下可检索的半成品版本。
- 可以按知识库、文档和元数据执行向量检索。

## 6. M3：基础流式 RAG

### 任务

- 实现 `ChatPipelineContext` 和阶段接口。
- 完成问题 → Embedding → pgvector → TopK → Prompt → LLM 的最小链路。
- 实现 SSE status/content/sources/error/done 事件。
- 实现请求 ID、全局超时、客户端断开和取消。
- 保存最小会话消息与 RAG Run。

### 验收

- 基于固定 Markdown 知识库完成端到端流式问答。
- 回答包含可追溯来源。
- 空检索不允许无依据编造答案。
- 客户端断开后模型 HTTP 和数据库资源能够释放。

## 7. M4：问题理解与记忆

### 任务

- 实现 Query Rewrite 与多问题拆分。
- 实现意图树加载、分类、阈值和全局检索兜底。
- 实现低置信或多意图歧义澄清。
- 实现最近 N 轮记忆与持久化摘要。
- 实现 Prompt 模板管理、引用编号和来源去重。
- 区分系统直答、知识库问答和澄清短路。

### 验收

- 多轮指代问题能够利用历史信息改写。
- 低置信问题能够澄清，而不是强行选择意图。
- 记忆摘要失败时可以降级到最近消息。
- Intent 20 条 Smoke Test 可以重复运行。

## 8. M5：检索后处理与评测

### 任务

- 实现内容/文档级去重。
- 实现加权 RRF 和三段检索预算校验。
- 接入 `qwen3-rerank` 或保持明确 No-op 降级。
- 运行 20 条 Smoke Evaluation。
- 运行 150 条完整 Evaluation。
- 完成 Rerank OFF/ON 单变量实验。
- 输出 Retrieval、Intent、Latency 和失败样本报告。
- 补齐 Docker、README、故障排查和发布检查。

### 验收

- 所有指标定义、样本数量和配置可追溯。
- 报告区分 Baseline、Variant 和控制变量。
- 不把参考项目或历史报告数字写成本项目结果。
- P0 范围内的关键失败场景均有测试。

## 9. P1：展示增强

P0 完成后按以下顺序评估：

1. MCP：订单查询、物流查询、售后资格检查，只读 Mock 工具优先。
2. 关键词检索：与向量结果进行 RRF 融合。
3. VLM：`qwen3.7-plus` 图片描述、图表解释和扫描件辅助。
4. XLSX：按 Sheet、表头和记录生成结构化 Chunk。
5. 模型路由：候选模型、首包探测、熔断与降级。
6. Embedding A/B：当前 BGE Base 与 BGE-M3 或云端模型比较。

每个 P1 能力独立建立 Issue、ADR 或实验记录，不批量打包进入主分支。

## 10. 开发过程中的决策点

| 时间点 | 必须确认 |
|---|---|
| M0 | Docker 端口、数据库名称、Python 环境和 CI 平台 |
| M1 | final/fast 模型 ID、超时和错误分类 |
| M2 | Chunk Token、Overlap、文件大小和解析策略 |
| M4 | 意图树结构、阈值和记忆窗口 |
| M5 | Workspace ID、Rerank 候选数和最终 TopK |
| P1 | MCP 工具范围、VLM 成本和 Embedding 是否切换 |

## 11. 发布前检查

- `git status` 无意外文件。
- `.env`、模型权重、数据库卷和个人文档未被跟踪。
- 安装、迁移、启动、测试和评测命令均在干净环境验证。
- README 只描述已实现能力，规划项明确标注。
- LICENSE、NOTICE 和第三方依赖许可证完整。
- 评测报告记录真实配置、失败样本和局限。
- GitHub Actions 或等价 CI 通过。
