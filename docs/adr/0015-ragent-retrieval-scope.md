# ADR-0015：复用 Ragent 的意图定向与全局检索作用域

- 状态：Accepted
- 日期：2026-09-01
- 取代：[ADR-0002](0002-document-index-schema.md)、[ADR-0005](0005-streaming-rag-api.md) 和
  [ADR-0009](0009-intent-routing-and-guidance.md) 中“调用方必须显式授权知识库 ID”的部分

## 背景

M4-C 为了建立最小三路 Baseline，把知识库范围交给聊天调用方，并要求每次请求携带非空知识库
ID。M5-D v4 证明：分类器看不到实际知识覆盖时，单靠静态 Prompt 无法稳定区分自营与第三方具体
商品。继续修改三路 Prompt 不能补上缺失的知识作用域机制。

Ragent 参考源码采用另一套机制：KB 意图节点绑定一个或多个 Collection；高置信 KB 意图可把检索
收窄到绑定 Collection；没有可用 KB 意图或置信度不足时，向量与关键词通道回退全部有效知识库。
系统类意图可以短路，空检索再返回无上下文结果。

本项目 P0/P1 不实现企业权限和多租户，因此不再用尚不存在的权限层改变参考项目的检索语义。

## 决策

1. 公开 `POST /api/v1/chat/stream` 不再接收 `knowledge_base_ids`，`scope` 整体可省略；保留的
   文档 ID、格式、解析器、章节和页码仅是可选数据库过滤条件。
2. 内部 `VectorSearchScope.knowledge_base_ids` 继续存在：非空表示高置信 KB 意图选择的定向范围，
   空元组表示全局范围。
3. 全局范围在一条 PostgreSQL/pgvector 查询中跨全部知识库召回，只纳入当前 Embedding model ID、
   revision、维度和归一化配置兼容且属于 active 文档版本的 Chunk。
4. 显式内部定向范围继续逐个验证知识库存在性和索引配置；不存在或不兼容返回原有稳定错误。
5. 分类器失败、超时或协议错误进入知识库全局兜底。明确歧义仍允许澄清；明确 SYSTEM 意图仍可
   跳过检索。
6. `rag_runs.knowledge_base_ids` 暂时保留用于兼容既有 Run；新公开聊天请求在启动 Run 时尚未完成
   Intent，因而写入空 UUID 数组。实际命中库仍可由持久化来源追溯，因此删除原有非空 Check
   Constraint；若后续需要单独记录“计划检索范围”，再以独立字段和迁移表达，避免混淆请求与结果。
7. 当前意图树的 KB 叶子可用稳定 slug 配置一个或多个知识库绑定；分类结果保留绑定，数据库
   Resolver 按配置顺序解析 UUID 后形成定向范围。未绑定叶子走本 ADR 的全局路径。

## 源码映射

- Ragent `IntentNode.getEffectiveCollectionNames()`：KB 意图到一个或多个 Collection 的绑定。
- Ragent `VectorSearchChannel.shouldNarrowToIntent()`：只有足够可靠的 KB 意图才收窄作用域。
- Ragent `VectorSearchChannel.retrieveGlobal()`：没有可用定向意图时跨全部有效 Collection 召回。
- Customer Agent 2 使用知识库 UUID 对应 Collection 身份，并用 SQL 的 active 版本及 Embedding
  身份条件表达“有效且兼容”。

## 后果

- 客户端调用更简单，不需要先枚举知识库 ID，分类失败也不再因调用方范围缺失而中断。
- v4 暴露的“模型不知道知识库覆盖范围”不再通过猜测解决；全局召回可以用真实文档是否命中决定
  是否存在上下文。
- 当前系统没有租户边界，所有知识库均属于同一个本地应用实例。未来引入多租户或企业权限时，
  必须在 `listActiveCollections` 的等价 Provider 中按可信身份过滤，而不是重新让请求体声明权限。
- 这是公开 API 和数据库核心约束变更，因此同步提供 Alembic 迁移、OpenAPI 测试和真实数据库验证。
