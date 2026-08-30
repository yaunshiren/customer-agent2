# ADR-0011：DashScope qwen3-rerank 与 20 条 OFF/ON Smoke

- 状态：Accepted
- 日期：2026-08-29
- 修订：[ADR-0010](0010-retrieval-post-processing-baseline.md)

## 背景

M5-A 已固定等权加权 RRF、内容去重、每文档最多 2 个 Chunk、Rerank 候选上限 40、最终
TopK 10，并以 `NoOpRerankModel` 建立可观察的 OFF Baseline。M5-B 只允许改变 Rerank 是否
启用，不能同时修改召回、融合或 TopK 参数。

用户已授权使用其本地配置的阿里云百炼 API Key 和 Workspace ID，执行固定 20 条真实
Rerank OFF/ON Smoke。真实凭据、Workspace ID、供应商原始错误和完整请求不得进入日志、报告或
Git。

## 决策

### 1. 供应商接口与请求

使用百炼 `qwen3-rerank` 专用 HTTP 接口，不通过 Chat 模型或 OpenAI Chat Completions 冒充：

```text
POST https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-api/v1/reranks
```

当前地域配置只接受 `cn-beijing`。请求包含 `model`、`query`、有序 `documents`、`top_n` 和
显式英文问答检索指令：

```text
Given a web search query, retrieve relevant passages that answer the query.
```

适配器使用已有 `httpx` 异步连接池，单次请求不自动重试。认证头只在内存中构造；异常、日志和
Trace 不保存 URL、Workspace ID、API Key、Query 或文档正文。

同一个 Workspace ID 和地域也用于构造 OpenAI-compatible Chat 专属地址：

```text
https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1
```

有 Workspace ID 时，应用的 final/fast Chat 与 Intent 评测统一使用该专属地址；没有 Workspace
ID 时，Chat 才回退到显式 `DASHSCOPE_BASE_URL`。这样可以避免业务空间 Key 被错误发往公共端点。

### 2. 响应与错误边界

成功响应必须包含顶层 `results`，每项提供唯一、未越界的原始索引和 0～1 有限相关性分数；
结果数量必须等于请求 `top_n`，分数必须按非升序排列。可选 `usage.total_tokens` 只用于评测成本
记录，不进入在线 RAG Trace。

已知欠费/额度错误码优先映射为额度耗尽，包括供应商以 HTTP 403 返回的
`AllocationQuota.FreeTierOnly`；其余 HTTP 401/403 映射为认证错误，408 和客户端超时映射为
超时，429 映射为限流，5xx 与网络错误映射为暂时不可用，其余不合规响应映射为协议错误。公开
错误文本保持稳定且脱敏。取消继续向上传播；应用关闭时释放 Rerank HTTP 连接池。

在线 Pipeline 仍使用 M5-A 的降级策略：已知模型错误、独立 10 秒超时或协议错误保留 RRF 顺序，
未知编程错误继续抛出。`RERANK_ENABLED=false` 仍是安全默认值；只有显式启用并同时配置 API Key
和 Workspace ID 时，默认服务图才构建真实适配器。

### 3. 20 条 OFF/ON Smoke

固定数据集包含 20 个合成客服问题，每个问题有 10 个有序候选和人工标注的相关候选 ID。数据不
来自用户文档或生产记录。OFF 直接使用相同候选输入顺序；ON 使用 `qwen3-rerank` 的返回顺序。

一次运行遵守以下约束：

- 必须显式传入 `--live` 才允许真实网络调用。
- 数据集必须恰好 20 条，按文件顺序串行执行。
- 每条 ON 最多一次请求，不重试，因此一次运行最多 20 次计费调用。
- 认证、配置、额度或协议等非重试错误会在首条失败后终止整轮，避免继续产生无意义调用；可重试型
  单样本失败在本次 Smoke 中也不重试，但继续记录其余样本。
- OFF 与 ON 使用完全相同的 Query、候选、`top_n=10` 和人工相关标注。
- 记录 Hit@1、Hit@3、MRR@10、逐样本相关候选首名次、胜/平/负、成功/失败数、成功请求延迟
  P50/P95 和供应商返回的总 Token。
- ON 失败按未命中计入全部 20 条的指标分母，同时单独记录稳定错误码，不能只报告成功子集。

Smoke 只证明适配器可用并提供首轮方向，不代表 150 条正式评测结论，也不能据此宣称生产效果
提升。报告必须标明合成数据、样本数、模型 ID、固定参数和该局限。

### 4. 变更边界

M5-B 不修改数据库模式、Embedding、向量召回、RRF、Prompt、Chat 模型或公开 SSE 字段；不新增
依赖。M5-C 再建立 150 条完整检索评测、失败样本分析和发布结论。

## 备选方案

### 使用 DashScope Python SDK

SDK 可以减少部分请求代码，但会新增依赖并隐藏 HTTP 资源、响应协议和错误映射。本项目已有
`httpx` 连接管理与 Mock 测试模式，直接实现小型专用适配器更容易验证。

### 并发运行 20 条样本

可以缩短耗时，但会增加瞬时限流与调用归因复杂度。Smoke 规模很小，串行、零重试更容易控制
调用次数和失败语义。

### 直接使用线上知识库问题

更接近真实流量，但会引入用户数据、权限与可提交性问题。M5-B 先使用合成固定集验证工程闭环；
正式数据集由 M5-C 单独治理。

## 后果

- 在线 Pipeline 可以在显式配置后使用专用云端 Rerank，并保持失败可降级。
- HTTP 正常、认证、额度、限流、超时、协议、取消和资源释放都有稳定测试锚点。
- 20 条真实调用有明确上限、指标和可复现报告格式。
- Smoke 结果只能解释为合成小样本实验，不能替代后续完整 Evaluation。

## 验证记录

2026-08-30 使用固定 `m5b-rerank-smoke-v1` 数据集完成一轮独立真实运行：

- 20 次请求全部成功，0 失败，未重试。
- OFF → ON：Hit@1 `0.10 → 1.00`，Hit@3 `0.30 → 1.00`，MRR@10
  `0.2928968254 → 1.00`。
- 胜/平/负为 18/2/0，总计 9,686 Token。
- 成功请求延迟 P50 为 196.685 ms，P95 为 231.871 ms。
- 报告不包含 API Key、Workspace ID、Query 或候选正文。

运行前曾遇到供应商以 HTTP 403 返回 `AllocationQuota.FreeTierOnly`。该事件暴露了错误映射顺序
问题，现已改为优先识别已知额度代码，并在首个非重试错误后终止整轮。最终指标仅说明专用适配器
和固定合成样本上的工程闭环，不作为生产效果声明。
