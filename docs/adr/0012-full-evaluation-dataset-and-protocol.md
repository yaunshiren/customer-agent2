# ADR-0012：原项目 150 条评测集与 M5-C 协议

- 状态：Accepted
- 日期：2026-08-30
- 关联：[ADR-0011](0011-dashscope-rerank-and-smoke-evaluation.md)

## 背景

M5-B 的 20 条合成 Smoke 只验证了专用 Rerank 适配器和调用边界，不能作为生产效果结论。
原项目附带 `eval_set_v1_all.jsonl`、20 条快速子集和 115 篇 Markdown 知识文档，用户明确授权
Customer Agent 2 使用这套资产完成 M5-C。

原始全量集有 150 条唯一 Query，其中 132 条要求检索、18 条明确不应检索；每条包含意图、难度、
期望答案类型、必需文档 ID、可选优选文档 ID、Ground Truth 和指标标签。115 篇正式文档之外，
3 条样本还引用 `_meta/product_mapping.md`，因此它以特殊文档 ID `PRODUCT_MAPPING` 参与语料库，
实际可检索文档数为 116。

## 决策

### 1. 数据快照和来源

在 `evaluation/datasets/ragenteval-v1/` 保存原始字节不变的版本化快照：150 条全量集、20 条子集、
115 篇正式知识文档和 `product_mapping.md`。不复制参考项目的运行结果、内部数据库 ID、脚本或
历史指标，也不修改 `ragenteval-main`。

加载器必须校验：150/20 条数量、各自 Query ID 唯一、20 个快速集 Query ID 均存在于全量集、
132/18 检索分布、116 个文档 ID 唯一、所有 required/nice 文档引用存在。标准文档 ID 从 Front
Matter 读取；唯一特殊映射固定为 `PRODUCT_MAPPING`。任何漂移直接失败，不能静默跳过。

20 条文件是原项目遗留的独立快速版本：它与全量集共用 Query ID，但部分 Query、文档标签或
Ground Truth 不同。M5-C 只以 150 条全量文件的记录值作为真值；20 条文件原样保留用于来源追溯，
加载时只校验 ID 子集关系，不合并或覆盖全量记录。

### 2. 语料库和检索范围

快照按原有四类建立四个评测知识库：product、manual、policy、faq；`PRODUCT_MAPPING` 放入
product。文档以业务 Doc ID 作为 `source_key`，因此检索结果可以直接与 Ground Truth 做
Doc 级比较，不依赖环境特定 UUID。

导入必须幂等：相同 Doc ID 和内容哈希已经 active 时跳过；内容变化时沿用正式入库用例创建新
版本。使用当前固定的 BGE Base 768 维、400/64 分块和 pgvector 配置，不为评测建立旁路索引。

### 3. 150 条评测协议

- Intent 分母为全部 150 条；132 条 `requires_rag=true` 映射 `knowledge_base`，18 条 no-rag
  映射 `system_direct`。本项目 P0 三路意图缺少明确澄清真值，因此不伪造 clarification 标签，
  也不报告该路由准确率。真实 Intent 串行、零重试，每条最多一次，完整一轮最多 150 次 fast
  Chat 计费调用；任何降级在首条发生时终止。
- Retrieval 分母只包含 132 条 `requires_rag=true` 样本；18 条 no-rag 样本用于过召回/路由
  检查，不能送入检索指标分母。
- required `expected_doc_ids` 是主真值；`expected_doc_ids_nice` 只做诊断，不提高主指标。
- Doc 级指标固定为 Hit@1/3/5/10、Recall@3/5/10、MRR@10，并记录 P50/P95、失败数和逐样本
  缺失文档。
- Rerank OFF/ON 必须复用同一次向量召回和融合候选。控制变量继续固定：向量召回 20、RRF
  `k=60`、每文档最多 2 个 Chunk、候选上限 40、最终 TopK 10。
- OFF 不访问云端；ON 对每个 RAG 样本最多一次专用 Rerank 请求，串行、零重试，完整一轮最多
  132 次计费调用。认证、配置、额度和协议错误在首条失败后终止。

报告只保存 Query ID、文档 ID、名次、指标、稳定错误码、延迟和 Token，不保存 API Key、
Workspace ID、Query、Ground Truth 或 Chunk 正文。任何云端 132 条 Rerank 或 150 条 Intent/Chat
运行必须显式参数开启，并在执行前单独获得用户对调用量和费用的确认。

### 4. 结论边界

参考项目的数据资产可以作为本项目输入，但参考项目历史报告和阈值不是 Customer Agent 2 的
实验结果。只有本项目代码、当前固定配置和本地/云端实际运行生成的报告可以进入 README 结论。

## 后果

- M5-C 有可提交、可追溯、严格校验的 150 条输入和116篇可检索文档。
- 评测复用正式解析、分块、Embedding、pgvector、RRF 和 Rerank 路径，不形成只为拿高分的旁路。
- 完整 ON 实验会产生最多132次真实 Rerank 费用，必须另行授权。
- 数据集没有明确 clarification 真值，三路 Intent 的 clarification 准确率仍是已知空白。
