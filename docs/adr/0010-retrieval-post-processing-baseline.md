# ADR-0010：M5-A 检索后处理与 No-op Rerank 基线

- 状态：Accepted
- 日期：2026-08-29
- 修订：[ADR-0005](0005-streaming-rag-api.md) 和
  [ADR-0008](0008-query-rewrite-and-multi-question-retrieval.md)；真实 Rerank 与 Smoke 由
  [ADR-0011](0011-dashscope-rerank-and-smoke-evaluation.md) 继续修订

## 背景

M4-B 已让最多三个子问题并发召回，并用“最好单查询名次、其次相似度”的临时规则合并候选。
该规则没有累积一个 Chunk 在多个子问题中重复命中的证据，也没有限制完全相同内容或单一文档
占满最终上下文。M1 虽已有类型化 `RerankModel` 和 `NoOpRerankModel`，在线 Pipeline 尚未调用，
因此 `RERANK_ENABLED=false` 目前没有可观察的在线降级行为。

用户已接受 M5 基线参数：RRF `k=60`、Rerank 候选上限 40、最终上下文 TopK 10，并先保持
Rerank 关闭。本阶段需要先建立可复现的 OFF Baseline；真实 `qwen3-rerank`、Workspace ID 和
ON/OFF 单变量实验在 M5-B 单独处理，避免同时改变融合与排序两个变量。

## 决策

### 1. 加权 RRF

每个子问题的向量结果视为一个有序结果列表。对同一 Chunk UUID 的融合分数为：

```text
score(chunk) = Σ weight(query) / (60 + rank(query, chunk))
```

实现接受显式正权重；当前只有同一向量通道的子问题，所有权重固定为 1.0。这样先验证 RRF 本身，
不在没有效果数据时人为偏置某个子问题。未来增加关键词或 MCP 通道时，必须通过评测决定不同
通道权重。并列时依次使用最好单查询名次、最高 Cosine 相似度、首次查询序号和 Chunk UUID，
保证结果可重复。

`k=60` 用于减弱第一名与后续名次的过大差异，同时保留多列表重复命中的累积优势；它是常见且
稳定的工程起点，不是本项目已经证明的最优值。

### 2. 三层重复控制与候选预算

融合按以下顺序处理：

1. 同一 Chunk UUID 在多个结果列表中合并分数。
2. 相同 `content_sha256` 只保留融合排名最好的代表，避免重复文本通过不同 Chunk ID 占位。
3. 每个 `document_id` 最多保留 2 个 Chunk，再截断到最多 40 个 Rerank 候选。

每文档 2 个 Chunk 是多样性安全基线：比“一文档只留一块”保留更多相邻或不同章节信息，同时
避免长文档占满 TopK 10。该参数和 RRF k 一样必须在后续检索评测中记录，不能描述为效果最优。

40 个候选位于最多 3 个子问题 × 每次 20 条召回的理论上限 60 与最终 TopK 10 之间，为后续专用
Rerank 留出足够选择空间，同时限制云端请求文本量和成本。Prompt Builder 仍保留 TopK 10 的
最终防线。

### 3. Rerank 编排与失败语义

RRF 后的候选通过 `RerankModel` 端口排序。当前默认服务图注入 `NoOpRerankModel`：返回融合顺序、
最多 10 条结果，并明确标记 `disabled`，不生成虚假的相关性分数。已知供应商模型错误、独立
Rerank 超时或返回索引/文档映射不合法时，也保留融合顺序并分别记录稳定降级原因；未知编程错误
继续向上抛出。

Rerank 独立超时采用 10 秒：40 个短候选的专用排序请求不应占用 120 秒全局预算的大部分。即使
发生降级，整个调用仍受现有 RAG 全局截止时间约束。M5-A 不新增外部调用或费用，不读取真实
Workspace ID。

### 4. Trace、SSE 与数据边界

Pipeline 在 `retrieving` 后增加 `fusing` 与 `reranking` 状态和 Trace。`fusing.candidate_count`
记录去重、文档上限和候选截断后的数量，`decision=weighted_rrf`；`reranking.candidate_count`
记录实际输出数量，`decision` 记录公开安全的 Rerank model ID，并在关闭、超时、供应商失败或
协议错误时记录稳定 `degradation_reason`。

Trace 不保存问题、候选正文、RRF 分数或 Rerank 分数。现有 `rag_runs.trace` 是 JSONB，因此无需
数据库迁移；公开 SSE 只增加阶段枚举，不改变已有字段含义。空融合结果继续使用 `no_context`
短路，不调用 Rerank 或最终 Chat 模型。

### 5. 本阶段边界

M5-A 不接入真实 `qwen3-rerank`，不请求 Workspace ID，不运行或编造 Rerank ON 的效果数字，
不增加关键词通道，也不开始 150 条完整评测。M5-B 在 OFF Baseline 固定后单独接入供应商适配器
并进行 ON/OFF 单变量实验。

## 备选方案

### 继续使用最好名次合并

改动最小，但不能奖励多个子问题共同命中的候选，也无法满足 P0 的 RRF 测试边界。

### 立即启用真实 Rerank

可以更快得到模型输出，但会同时改变融合算法和排序模型，无法解释结果差异，也需要提前引入
Workspace ID 与调用成本。当前先固定 OFF Baseline。

### 每个文档只保留一个 Chunk

多样性最强，但可能丢失同一长文档中互补的两个相关章节。每文档最多两个是更保守的首轮基线。

## 后果

- 多子问题重复命中的 Chunk 获得可解释的累计排名优势。
- 重复内容和单文档过度占位在调用 Rerank 前被有界处理。
- Rerank 关闭或已知失败不再是隐式行为，日志、Trace 和测试都能识别。
- M5-B 可以只改变 Rerank OFF/ON 一个变量，并复用完全相同的 RRF 候选。
- 当前参数仍需固定评测集验证，README 和报告不得声称相关性已经提升。
