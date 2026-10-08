# GA4 电商增长诊断与实验评估

从电商行为数据出发，回答三个问题：**用户在哪一步流失？什么问题值得优先改？怎样判断改动带来了增长？**

项目覆盖 92 天、429 万条事件，建立 36 万条会话事实，贯通漏斗与留存分析、数据质量诊断、结算机会识别、实验设计与因果估计验证。

**核心判断：优先调查结算个人信息页到支付资料页之间的流失。** 商品联接识别出 646 名用户集中于同一 SKU 的异常记录；排除该组合和重新设定入组窗口后，支付前路径仍占七日未购买者的 58%–59%。据此提出先核查集中异常及步骤记录，再验证全设备个人信息页简化的方案。

## 结果一览

| 工作 | 主要结果 | 入口 |
|---|---|---|
| 数据开发 | 4,295,584 条事件，360,129 个会话，5,692 条购买事件，241,261 条商品日记录 | [数据与 SQL](docs/data.md) |
| 漏斗和留存 | 严格顺序会话漏斗、D0–D14 exact-day 留存、设备/来源拆解、92 日日度指标 | [分析方法与结果](docs/analysis.md) |
| 结算诊断 | 4,872 个首次合格 checkout 用户，2,364 个七日购买用户，基线 48.52% | [用户基线结果](results/validation/baseline_comparison.csv) |
| 异常与敏感性 | 集中异常 646 人；支付前未购占比从 68.86% 调整为 58.06%，晚期重新入组为 59.28% | [敏感性结果](results/validation/review_sensitivity_baselines.csv) |
| 实验规划 | 检测 +5 个百分点，80% power 需 3,136 人；按保守历史流量约 11 周 | [实验方案](docs/experiment.md) |
| 统计验证 | 500 次历史 A/A：误报率 5.4%，95% 区间覆盖率 94.6%；重复会话错当独立样本时模拟误报升至 28.0% | [验证汇总](results/validation/historical_aa_replay_summary.csv) |
| 因果方法 | 比较 OR、IPW、交叉拟合 AIPW；检验模型错设、未测混杂与重叠不足 | [方法与解释](docs/causal.md) |

![结算异常敏感性](results/analysis/checkout_basket_sensitivity.png)

数据来自 Google Merchandise Store 的 **2020-11-01 至 2021-01-31 公开混淆样本**。项目成果是数据分析、实验方案和离线方法验证；产品 A/B 为设计方案，模拟效应与真实观察结果分别呈现。

## 阅读顺序

- **快速了解结论**：[分析报告](docs/analysis.md)。
- **查看数据建模与 SQL**：[数据说明](docs/data.md) → [会话事实 SQL](sql/01_session_facts.sql)。
- **查看统计与实验能力**：[实验设计](docs/experiment.md) → [估计方法](src/methods.py) → [因果验证](docs/causal.md)。
- **复现结果**：运行下方命令；`results/report.html` 为生成的本地浏览器报告。

![漏斗与留存](results/overview/funnel_retention.png)

## 复现

需要 Python 3.11 或更高版本。仓库附带压缩 CSV 输入，完整分析可离线运行，不需要 Google 账号。

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

程序依次完成 20 项数据检查、基础分析、交易诊断、用户基线、敏感性、用户聚类 bootstrap、A/A、半合成 RCT 与因果验证，最后生成报告。完整模拟每个 RCT 场景 500 次、每个因果场景 300 次，每次 6,000 人；运行时间取决于机器。

```bash
python run.py --stage data         # 解压核心输入并检查粒度与对账
python run.py --stage analysis     # 真实数据分析，包含全部前置步骤
python run.py --stage validation   # 重建分析输入并运行统计验证
python src/checks.py               # 时间边界、数据泄漏和估计量逻辑检查
```

`data/processed/` 是自动生成的 CSV 工作文件；`results/` 保留供阅读的核心表和图，其余逐次模拟结果运行后保存到 `data/processed/details/`。所有随机验证使用固定种子。已保存的输入从 SQL 聚合结果开始；重新查询源事件的方法见[数据说明](docs/data.md#重新查询公开数据)。

## 结构

```text
├── README.md
├── requirements.txt
├── run.py                  # 统一运行入口
├── sql/                    # 10 个源查询：事实表、商品、页面、留存、日度
├── src/                    # Python 分析、统计方法与验证
├── data/inputs/             # 10 份压缩 CSV，约 26 MB
├── docs/                   # 数据、分析、实验、因果方法
└── results/                # 核心结果表、图与可生成的 HTML 报告
```

## 技术要点

- **SQL**：嵌套参数提取、CTE、窗口函数、数组处理、逐级后继事件匹配、条件聚合与多粒度事实表。
- **Python**：数据联接与去重、向量化用户窗口、分组统计、敏感性分析与结果可视化。
- **统计推断**：比例区间、用户聚类 bootstrap、均值差标准误、功效分析、Monte Carlo 误差。
- **实验设计**：用户级 sticky 分流、触发前入组、ITT、七日成熟窗口、SRM、收入护栏与决策门槛。
- **因果推断**：潜在结果、可交换性与 positivity、IPW、AIPW、交叉拟合、ESS/SMD 与裁剪敏感性。

## 数据来源与参考

[Google 公开样本说明](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset) · [GA4 导出字段](https://support.google.com/analytics/answer/7029846) · [Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060)

输入数据保持 Google 公开样本的来源属性；匿名 ID 表示浏览器标识，渠道字段为 first-user source。测量诊断系统处理占位值、交易重复与记录漂移，提升分析结果的可靠性。
