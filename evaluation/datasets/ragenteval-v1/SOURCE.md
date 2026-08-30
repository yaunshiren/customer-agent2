# ragenteval-v1 数据来源

本目录是原项目附带评测资产的字节级快照，用于 Customer Agent 2 的 M5-C 独立复现。

- 来源仓库目录：`ragenteval-main/ragenteval-main`
- 全量集：`eval/rag/dataset/eval_set_v1_all.jsonl`（150 条）
- 遗留快速集：`eval/rag/dataset/eval_set_v1.jsonl`（20 条）
- 知识文档：`knowledge_base/01_product` 至 `04_faq`（115 篇）
- 特殊检索文档：`knowledge_base/_meta/product_mapping.md`（Doc ID `PRODUCT_MAPPING`）
- 快照日期：2026-08-30

原始文件在复制时逐文件校验 SHA-256，118 个文件均与来源一致。`SOURCE.md` 是本项目新增的来源
说明，不属于原始数据。Customer Agent 2 不使用参考项目的历史运行结果或指标；所有公开结果必须
由本项目代码重新运行产生。

20 条遗留快速集的 Query ID 均属于全量集，但其中 10 条与全量记录存在字段差异（1 条 Query、
1 条 required 文档标签、9 条 Ground Truth，部分差异发生在同一条记录）。因此 M5-C 以 150 条
全量集为唯一真值，快速集仅保留作来源追溯，不用于覆盖全量记录。
