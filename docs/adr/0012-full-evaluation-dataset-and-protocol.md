# ADR-0012：原项目 150 条评测集与 M5-C 协议

- 状态：Accepted
- 日期：2026-08-30
- 关联：[ADR-0011](0011-dashscope-rerank-and-smoke-evaluation.md) 和
  [ADR-0013](0013-intent-failure-calibration.md)

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
  Chat 计费调用；任何降级在首条发生时终止。每条成功后原子写入脱敏 checkpoint；恢复时必须
  使用完全相同的数据集、模型、阈值、超时、Token 和思考模式，并只允许连续成功前缀。每次运行
  的付费确认数必须精确等于 checkpoint 剩余条数。
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
- 完整 ON 实验已在独立授权下完成 132 次真实 Rerank 调用；后续复跑仍必须另行授权。
- 数据集没有明确 clarification 真值，三路 Intent 的 clarification 准确率仍是已知空白。

## 验证记录

2026-08-30 使用固定 `ragenteval-v1` 快照完成 132 条 Retrieval OFF/ON：

- 本地 OFF 与真实 ON 的检索调用均覆盖 132 条；ON 132/132 次 `qwen3-rerank` 成功、0 失败、
  零重试，共 210,769 Token，延迟 P50/P95 为 199.335/266.688 ms。
- OFF → ON 的 Hit@1 为 `0.7045 → 0.6364`、Hit@3 为 `0.8939 → 0.8712`、Hit@5 为
  `0.9545 → 0.9470`、Hit@10 为 `0.9848 → 0.9773`。
- OFF → ON 的 Recall@3 为 `0.6723 → 0.6578`、Recall@5 为 `0.7683 → 0.7689`、
  Recall@10 为 `0.8447 → 0.8327`，MRR@10 为 `0.8075 → 0.7622`，胜/平/负为 21/81/30。
- 该配置下 Rerank 没有改善完整集，默认继续关闭；结果不外推到其他模型、Prompt、候选预算或
  数据集。
- 指标按前 10 个 Chunk 截断后再做 Doc 去重。付费 ON 结束后修正了旧 OFF 辅助统计继续扫描
  Top10 以外 Chunk 的问题；ON 原始结果保留，OFF 只做确定性本地复算，没有追加云端请求。

同日完成最终独立 150 条真实 Intent：

- `qwen3.8-flash` 150/150 成功、0 降级、零重试；温度 0、关闭思考模式、阈值 0.75/0.10、
  最大输出 256 Token。评测专用超时 60 秒，线上默认仍为 20 秒。
- 总体准确率 `128/150 = 85.33%`；RAG 为 `121/132 = 91.67%`，no-rag 为
  `7/18 = 38.89%`；SUPPORT/FEEDBACK/CHAT 为 `92.80%/33.33%/70.00%`。
- system_direct 真值的 18 条中，7 条正确、5 条误进 knowledge_base、6 条转 clarification；
  knowledge_base 真值的 132 条中，121 条正确、11 条转 clarification。
- 输入/输出 Token 为 45,101/7,366，延迟 P50/P95 为 794.040/2,459.129 ms。
- 最终报告只统计这轮独立 150 条。此前调试过程累计包含 2 次 `qwen3.7-flash` 首条失败，以及
  一轮 `qwen3.8-flash` 在第 145 条触发 20 秒超时；整个获批过程累计 297 次尝试。旧运行器未
  持久化中间成功结果的问题已由 checkpoint 修复，历史调试 Token 不混入最终报告。
