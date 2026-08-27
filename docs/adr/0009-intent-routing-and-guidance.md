# ADR-0009：三类 Intent 路由与澄清事件

- 状态：Accepted
- 日期：2026-08-27
- 修订：[ADR-0005](0005-streaming-rag-api.md) 和
  [ADR-0006](0006-conversation-rag-run-persistence.md)

## 背景

M4-A/M4-B 已让历史记忆参与问题改写，并让最多三个子问题实际参与检索，但所有请求仍无条件进入
知识库召回。问候、能力说明等系统问题不应浪费向量检索；语义不完整或两个路由分数接近的问题也
不应被强行回答。ADR-0007 已由用户确认三类初始路由、0.75 高置信阈值和 0.10 歧义差值，M4-C
需要把这些参数接入可测试、可持久化、可降级的在线 Pipeline。

## 决策

### 1. 配置化意图树

默认安装包包含版本化 JSON 意图树，初始只允许三个叶子：

- `system_direct`：问候、系统能力和使用方式，不需要知识事实。
- `knowledge_base`：需要在请求已授权的知识库作用域中查找事实并引用。
- `clarification`：问题缺少决定性信息，需要先向用户提问。

启动时加载并严格验证版本、路由集合、唯一名称和非空描述；无效配置快速失败。当前树不虚构订单、
物流、售后工具等 P1 业务意图，也不根据自由文本扩大知识库权限。

### 2. 严格分类与阈值

Query Rewrite 成功或降级后，复用 fast Chat 模型输出严格 JSON：三个路由各一个独立的 0～1
置信分数，并包含可空的澄清问题。分数不强制归一化，确保高置信 Top1 与接近的 Top2 可以同时
存在，0.10 歧义差值才具有实际决策空间。服务端按固定顺序稳定排序：

1. Top1 小于 0.75，转为 `clarification`，原因是 `low_confidence`。
2. Top1 与 Top2 差值小于 0.10，转为 `clarification`，原因是 `ambiguous`。
3. Top1 为 `clarification`，直接澄清，原因是 `explicit_clarification`。
4. 其余高置信结果进入 `system_direct` 或 `knowledge_base`。

边界使用“达到阈值即可自动路由、达到 0.10 即不再视为接近”，便于固定样本精确复现。阈值是
ADR-0007 的工程 Baseline，不代表已经通过真实模型评测证明最优。

分类默认独立超时 20 秒、最多输出 256 token：20 秒限制 fast 模型故障对 120 秒全局预算的占用；
三个分数和一个短问题无需 512 token，256 可降低不必要的输出成本。参数均可配置且继续受 RAG
全局截止时间约束。

### 3. 失败降级与作用域

已知模型失败、独立超时或协议不合规时，不让辅助分类能力中断原本可用的 RAG，而是降级到
`knowledge_base`，使用调用方请求中已经显式授权的完整知识库作用域。该行为称为“授权范围内的
全局检索兜底”，绝不等于搜索未授权知识库。Trace 和结构化日志记录稳定降级代码，不记录问题、
Prompt、分数原文或异常详情。未知编程错误仍向上抛出。

### 4. 三路 Pipeline 语义

- `knowledge_base`：继续执行 M4-B 并发向量检索、Prompt、引用和最终模型流。
- `system_direct`：跳过向量检索，使用 final 模型和限制能力边界的安全 Prompt 流式回答；不产生
  `sources`，不得声称访问订单、物流、联网信息或知识库事实。
- `clarification`：不执行检索和最终模型流，输出一个 `guidance` 事件和
  `outcome=clarification` 的 done。澄清文本保存为 assistant 消息，使用户下一轮的短回答能与
  问题组成完整记忆。

v1 请求仍要求显式非空知识库作用域，即使最终被路由为系统直答；这样不在 M4-C 同时修改请求
授权契约。后续若允许无知识库的独立系统请求，需要单独版本化 API 决策。

### 5. SSE、Trace 与持久化

Pipeline 增加 `intent` 和 `clarification` 状态。新增 `guidance` 事件，包含非空 `message` 与
`low_confidence`、`ambiguous` 或 `explicit_clarification` 原因。done 增加 `intent_route`，并增加
`clarification` outcome。Trace 增加可选稳定 `decision`；Intent Trace 保存最终路由、候选数量和
可选降级代码，不保存分数或用户正文。

数据库迁移为 `rag_runs` 增加可空 `intent_route` 并允许 `clarification` 状态。已有
completed/no_context Run 回填为 `knowledge_base`；failed/cancelled/running 保持空路由。系统直答
以 completed 保存 assistant 消息但来源数组为空；知识库 completed 仍必须有来源；clarification
保存 guidance assistant 消息、fast 模型结果和 Trace。最近记忆与摘要把 completed 和
clarification 都视为完整 user/assistant 轮次。

### 6. Smoke Test 边界

仓库固定 20 条合成问题、模型结构化输出和期望决策，离线验证三路解析、0.75/0.10 边界以及低
置信/歧义澄清。该 Smoke Test 证明协议和决策规则可重复，不是对真实 `qwen3.7-flash` 的准确率
报告；真实 Intent Top-1、延迟和失败样本仍需后续固定模型评测。

本阶段不实现业务域知识库映射、权限系统、RRF、Rerank、MCP 工具路由或 150 条完整评测。

## 备选方案

### 分类失败直接返回错误

可保持严格语义，但会让一个辅助 fast 模型故障破坏已有知识库问答。授权范围内检索兜底能保留
可用性，同时不会扩大数据访问范围。

### 低置信时直接全局检索并回答

表面成功率更高，但用户语义不明确时仍可能基于错误问题生成有引用的错误答案。当前优先显式
澄清；只有分类器自身故障才采用知识库兜底。

### 将 guidance 当作 no_context

无需数据库迁移，但会混淆“知识库没有资料”和“用户问题不明确”，也无法保存澄清消息供下一轮
记忆使用。因此增加独立终局。

## 后果

- 问候与系统能力问题不再访问 pgvector，知识事实问题继续保持引用约束。
- 低置信和歧义问题产生明确、可持久化的澄清，而不是强行选择。
- fast 分类故障会增加一次可观察的知识库兜底，而不会静默扩大权限。
- `guidance`、`intent_route` 和 `clarification` 是公开 SSE 的兼容增量；严格枚举客户端需要同步。
- M4-C 的 20 条样本只能作为工程 Smoke，进入效果结论前必须运行真实固定模型评测。
