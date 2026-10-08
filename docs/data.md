# 数据建模与 SQL

数据开发的目标是把事件流转换为可以用于决策的分析单位：会话用于路径诊断，购买事件用于交易质量检查，首次合格用户用于实验设计。三种分母分别定义与使用。

## 输入表与来源

源数据集为 `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`，覆盖 2020-11-01 至 2021-01-31。文件均在 `data/inputs/`，以 `.csv.gz` 保存；解压后的同名 CSV 由 `src/data.py` 生成。

| 输入 | 粒度 | 行数 | 查询 |
|---|---|---:|---|
| event_inventory | 日期 × 事件 × 设备 | 4,143 | `00_event_inventory.sql` |
| session_facts | user_pseudo_id × ga_session_id | 360,129 | `01_session_facts.sql` |
| transaction_events | 购买事件 | 5,692 | `02_transaction_events.sql` |
| item_daily | 日期 × 设备 × SKU × 商品标签 | 241,261 | `03_item_daily.sql` |
| event_semantics_profile | 日期 × 事件 × 页面 × 商品数组长度 | 28,691 | `04_event_semantics.sql` |
| checkout_index_facts | 会话首次 checkout 的篮子 | 11,106 | `05_checkout_baskets.sql` |
| checkout_local_price | 会话首次 checkout 的原生价格 | 11,106 | `06_checkout_prices.sql` |
| price_coverage | 事件类型的价格和货币覆盖 | 3 | `07_price_coverage.sql` |
| retention_cohorts | 首访 cohort 日 × 第 n 日 | 450 | `08_retention.sql` |
| daily_metrics | 日期，July 阶段保存的日度结果 | 92 | `09_daily_metrics.sql` |

这些文件是从完整源窗口计算的事实表和聚合导出，不是 429 万条原始事件的逐条副本。保留完整会话和购买数据，足以重新构造实验用户基线、关联商品篮子，并重跑统计验证。

## 会话事实

会话键采用 `(user_pseudo_id, ga_session_id)`。session ID 本身不是全局唯一键；跨日会话按同一键聚合，`session_date` 为最早 property date，`session_last_date` 记录末日。

1. `extracted` 使用相关子查询读取 `event_params`，避免同时展开参数和商品造成笛卡尔积。
2. `valid` 保留匿名 ID 存在、且恰有一个有效 session 参数的事件。
3. `sessions` 聚合会话边界、属性、事件计数和原始事件数组。
4. `stages1`、`stages2` 逐级寻找时间严格大于上一阶段的最早事件。
5. 输出固定粒度事实表。收入只累计 purchase 事件；商品展开在独立查询中处理。

例如 `view → checkout → purchase` 要求存在 `t_view < t_checkout < t_purchase`。分别比较三个事件类型的全局最早时间会漏掉回访路径；逐级找后继事件可以正确处理 `checkout → view → checkout → purchase`。同时间戳采用严格 `>`，同时保留少量弱顺序指标用于诊断。

## 购买与商品

`transaction_events` 保留购买金额和完整 `items_json`。`event_id` 只是本地行标识，用来联接计算结果，不是后台订单号。SQL 对完全相同的记录计数，并保留其出现次数；已保存输入的每个购买记录出现一次。

交易研究用“金额为正，且至少有一个有效 SKU、正数量商品”定义 **positive-merchandise purchase**。重复候选由同用户、同会话、同交易 ID、同金额及排序后同篮子共同确定。保留原始记录，并报告折叠敏感性，避免仅凭混淆的 `transaction_id` 删单。

`item_daily` 用于商品构成与字段诊断。SKU 在不同事件类型中的命名空间不完全一致，因此先通过标识一致性检查，再选择具有可比性的分析单位。

## 时间与指标

| 口径 | 窗口 | 目的 |
|---|---|---|
| 完整源窗口 | 2020-11-01 至 2021-01-31 | 盘点与漂移诊断 |
| 会话稳定观察窗 | session_date 为 2020-11-30 至 2021-01-25 | 路径和设备比较 |
| 主实验规划入组窗 | UTC [2020-11-30, 2021-01-19) | 每个匿名 ID 取窗口内首次 checkout |
| 结果成熟截止 | UTC 2021-01-26 之前 | 每人完整 7×24 小时购买标签 |
| 事前历史 | [index−14d, index)，且会话已在 index 前结束 | 防止使用入组后的行为 |
| 留存 cohort | 11 月有 first_visit 的用户 | exact calendar-day D0–D14 留存 |

`event_date` 按 GA4 property 时区记日；微秒时间戳按 UTC 计算，二者分别使用。没有历史记录表示没有可观察历史，不自动定义为新客。设备/来源描述使用 first-user source，不作为当次访问渠道归因。

## 数据检查

`src/data.py` 检查输入行数、会话键唯一、时间顺序、事件总数与购买事件对账、商品金额总额、篮子键、留存分母及 D0。`src/checks.py` 用有明确预期的微型数据测试严格窗口、特征泄漏和用户重复问题。

`results/data_checks.csv` 保存数据检查结果。业务研究进一步检查零金额、SKU 标签、交易 ID 冲突、事件时序和记录随时间的变化。总额对账与逐事件差异检查共同覆盖数量和金额质量。

## 重新查询公开数据

在自己的 BigQuery 项目中逐个执行 `sql/00` 至 `sql/09`，源表均为公开数据集，没有绑定个人云项目。查询完整窗口后导出 CSV，按上表命名压缩并放入 `data/inputs/`。大结果使用 BigQuery 导出功能或按已经聚合后的日期分片，合并时检查主键与总行数，不从浏览器预览中截取部分行。

`02_transaction_events.sql` 的 `event_id` 是确定性排序编号；`05_checkout_baskets.sql` 的同时间戳记录用序列化内容做稳定选择并报告变体数。公开版去除了存储元数据。`src/data.py` 中的行数针对仓库附带的固定样本；替换输入后应重新核对并调整预期值。

本次整理验证的是附带 CSV 的离线完整复现；SQL 目录提供重新提取的方法，整理期间未重新执行云端查询。

字段定义参见 [GA4 BigQuery Export schema](https://support.google.com/analytics/answer/7029846)；数据范围及混淆特性参见 [Google 样本说明](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset)。
