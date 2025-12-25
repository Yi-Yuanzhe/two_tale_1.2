import pandas as pd
import os

print("=== 诊断脚本开始 ===")

# 1. 检查合并后的 COT 源文件
cot_file = 'data/cftc_legacy/legacy_cot_data.csv'
if os.path.exists(cot_file):
    df_cot = pd.read_csv(cot_file, low_memory=False)
    print(f"\n[1] COT 源文件总行数: {len(df_cot)}")
    
    # 检查日期列
    date_cols = [c for c in df_cot.columns if 'date' in c.lower()]
    print(f"    发现可能的日期列: {date_cols}")
    if date_cols:
        # 尝试查看最早和最晚的日期（字符串格式）
        dates = df_cot[date_cols[0]].sort_values()
        print(f"    日期范围 (原始字符串): {dates.iloc[0]} 到 {dates.iloc[-1]}")

    # 检查代码列类型
    code_col = [c for c in df_cot.columns if 'Code' in c and 'CFTC' in c]
    if code_col:
        print(f"    代码列 ({code_col[0]}) 的第一个值: '{df_cot[code_col[0]].iloc[0]}'")
        print(f"    代码列的数据类型: {df_cot[code_col[0]].dtype}")
else:
    print("\n[!] 错误: COT 源文件不存在！")

# 2. 检查一个完整的价格文件
price_file = 'data/prices/CL_prices.csv'  # 以原油为例
if os.path.exists(price_file):
    df_price = pd.read_csv(price_file)
    print(f"\n[2] 原油价格文件行数: {len(df_price)}")
    print(f"    日期范围: {df_price['Date'].min()} 到 {df_price['Date'].max()}")
    
    # 模拟重采样
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_weekly = df_price.set_index('Date').resample('W-TUE').last()
    print(f"    重采样后的周数: {len(df_weekly)}")
else:
    print("\n[!] 错误: 价格文件不存在！")

print("\n=== 诊断建议 ===")
print("如果 [1] 的行数很少 -> 问题在 COT 下载或解析")
print("如果 [1] 的代码列是 int64 -> 问题在代码匹配 (前导零丢失)")
print("如果 [1] 和 [2] 都很大，但合并后很小 -> 问题在日期对齐 (Merge)")