# A Tale of Two Premiums - 复现项目

论文 **"A Tale of Two Premiums: The Role of Hedgers and Speculators in Commodity Futures Markets"** 的 Python 复现。

原文链接：
- [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12845)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2449315)

参考笔记：
- https://xseeope.github.io/#/papers/Futures/two_premium
## 项目简介

本项目复现了论文中关于套期保值压力、交易者持仓与商品期货收益率之间关系的实证分析，涵盖 22 个主要商品市场。

## 快速开始

```bash
# 安装依赖
pip install pandas numpy yfinance requests statsmodels scipy

# 下载数据
python data_acquisition.py

# 数据预处理
python data_preprocessing.py

# 运行回归分析
python table_replication.py
```

## 项目结构

```
├── data/
│   ├── cftc_legacy/        # CFTC 持仓报告
│   ├── prices/             # 商品期货价格数据（Yahoo Finance）
│   └── processed/          # 合并后的周频数据
├── output/
│   └── tables/             # 回归结果
├── data_acquisition.py     # 下载 CFTC 持仓数据和价格数据
├── data_preprocessing.py   # 计算变量并对齐时间序列
└── table_replication.py    # Fama-MacBeth 回归分析
```

## 数据来源

**CFTC 持仓数据** (1994-2017)
- 每周 Commitment of Traders (COT) 报告
- 商业交易者 vs. 非商业交易者持仓
- 数据源：https://www.cftc.gov/MarketReports/CommitmentsofTraders/

**商品价格数据** (1994-2017)
- 22 个商品期货合约，涵盖能源、金属、谷物、软商品、畜牧
- 日频价格转换为周频（周二收盘价）
- 数据源：Yahoo Finance

**涵盖商品**
> [!note]与原文中相比仅缺少 Minneapolis HRS (Hard Red Spring) Wheat。

| 类别   | 代码 |
|--------|------|
| 能源   | CL, HO, NG, RB |
| 金属   | GC, SI, HG, PL, PA |
| 谷物   | ZW, ZC, ZS, ZL, ZM, ZO, KE |
| 软商品 | KC, SB, CC, CT, OJ |
| 畜牧   | LE, HE, GF |

## 核心变量

遵循论文方法构造的变量：

**套期保值压力 (HP)**
```
HP = (商业交易者空头 - 商业交易者多头) / 未平仓合约数
```

**净交易量 (Q)**
```
Q = Δ(净多头持仓) / OI_{t-1} × 100
```
净持仓变化的百分比（商业 vs. 非商业交易者）

**交易倾向 (PT)**
```
PT = (|Δ多头| + |Δ空头|) / (多头 + 空头)
```

**特质波动率 (v_t)**
```
v_t = annualized std(商品收益率对 S&P 500 回归的残差)
```
52 周滚动窗口计算

**超额收益率**
```
Ret_{t+1} = (F_{t+1} - F_t) / F_t
```

## 回归分析

本项目实现了 Fama-MacBeth 横截面回归：

**Table I**：收益率、套期保值压力和交易活动的描述性统计

**Table II**：持仓变化与收益率
- 商业交易者对价格变化的响应
- 非商业交易者的动量交易行为

**Table III**：收益率可预测性
- 套期保值压力溢价
- 净交易量效应
- 特质波动率交互作用

**Table IV**：风险溢价分解
- 套期保值溢价 vs. 投机溢价
- 控制变量稳健性检验

## 输出结果

结果保存在 `output/tables/` 目录：
- `table_I_summary_statistics.csv`
- `table_II_position_changes_returns.csv`
- `table_III_return_predictability.csv`
- `table_IV_premium_decomposition.csv`

每个文件包含回归系数、t 统计量和 p 值。

## 注意事项

- **时间对齐**：COT 报告（周二收盘）与周频价格数据匹配
- **连续合约**：使用 Yahoo Finance 连续合约作为主力合约的代理
<!-- - **数据期间**：1994-2017（与原始论文一致） -->
- **基差数据缺失**：真实基差（近月-远月价差）无法从免费数据源获取，使用简化代理

## 局限性

- Yahoo Finance 提供连续合约，非特定到期月份合约
- 真实基差计算需要期货曲线数据（免费数据源无法获取）
- 部分商品早期数据有限
- CFTC 细分数据（disaggregated data）仅从 2006 年开始提供

---

**说明**：本项目为使用公开数据的独立复现。由于数据源差异，结果可能与原始论文存在偏差。
