# TODO

## 高优先

找到提供全合约的数据源并重新构造变量
- 目前使用的 yfinance 接口只能获取主力合约，而原文中被解释的收益率溢价为近月合约的横截面溢价。同时文章中使用了基差作为控制变量，没有全体合约信息也无法构造基差变量。

## 次优先

稳健性检验：
- 使用 DCOT 持仓数据复现：Table IV，Table VII
- 文章非主要结论的 Table 复现：Table IX，Table X，Table XI，Table XII

>[!note] 复现新 table 时尽量尝试新开 py 文件，不要修改原 py 文件，保证可读性。

