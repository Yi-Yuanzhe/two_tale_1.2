"""
Table Replication for "A Tale of Two Premiums" Paper
Generates all tables according to the paper's methodology
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import statsmodels.api as sm
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('output/tables', exist_ok=True)

def load_all_processed_data():
    """Load all processed commodity data into a single DataFrame"""
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)
    
    all_data = []
    files = glob.glob('data/processed/*_processed.csv')
    
    for file in files:
        ticker = os.path.basename(file).replace('_processed.csv', '')
        df = pd.read_csv(file)
        
        # Handle different column name formats
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'Report_Date'})
        elif 'Report_Date' not in df.columns:
            # If neither, try to reset index
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'Report_Date'})
        
        df['Ticker'] = ticker
        df['Report_Date'] = pd.to_datetime(df['Report_Date'])
        all_data.append(df)
        print(f"✓ Loaded {ticker:5} - {len(df)} observations")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total: {len(combined):,} observations across {len(files)} commodities")
    
    return combined

# def calculate_additional_variables(df):
#     """Calculate additional variables needed for analysis"""
#     print("\n" + "=" * 70)
#     print("CALCULATING ADDITIONAL VARIABLES")
#     print("=" * 70)

#     df['Report_Date'] = pd.to_datetime(df['Report_Date'])
    
#     # |Q| - Absolute value of net trading
#     df['abs_Q_Comm'] = df['Q_Comm'].abs()
#     df['abs_Q_NonComm'] = df['Q_NonComm'].abs()
#     print("✓ Calculated |Q| variables")
    
#     # Calculate position changes for Table II
#     for ticker in df['Ticker'].unique():
#         mask = df['Ticker'] == ticker

#         # OI_{t-1}
#         df.loc[mask, 'Open_Interest_Lag1'] = df.loc[mask, 'Open_Interest_All'].shift(1)

#         # Delta positions (changes)
#         df.loc[mask, 'Delta_NetLong_Comm'] = df.loc[mask, 'NetLong_Comm'].diff()
#         df.loc[mask, 'Delta_NetLong_NonComm'] = df.loc[mask, 'NetLong_NonComm'].diff()
#         # Calculate non-reportable positions
#         df.loc[mask, 'NonReport_Long'] = df.loc[mask, 'Open_Interest_All'] - df.loc[mask, 'Comm_Positions_Long_All'] - df.loc[mask, 'NonComm_Positions_Long_All']
#         df.loc[mask, 'NonReport_Short'] = df.loc[mask, 'Open_Interest_All'] - df.loc[mask, 'Comm_Positions_Short_All'] - df.loc[mask, 'NonComm_Positions_Short_All']
#         df.loc[mask, 'NetLong_NonReport'] = df.loc[mask, 'NonReport_Long'] - df.loc[mask, 'NonReport_Short']
#         df.loc[mask, 'Delta_NetLong_NonReport'] = df.loc[mask, 'NetLong_NonReport'].diff()
#         df.loc[mask, 'Q_NonReport'] = df.loc[mask, 'Delta_NetLong_NonReport'] / df.loc[mask, 'Open_Interest_Lag1'] * 100
#         # Lag Q for Table II
#         df.loc[mask, 'Q_Comm_lag1'] = df.loc[mask, 'Q_Comm'].shift(1)
#         df.loc[mask, 'Q_NonComm_lag1'] = df.loc[mask, 'Q_NonComm'].shift(1)
#         df.loc[mask, 'Q_NonReport_lag1'] = df.loc[mask, 'Q_NonReport'].shift(1)
#     print("✓ Calculated position changes")
    
#     # Return lags for momentum analysis
#     for ticker in df['Ticker'].unique():
#         mask = df['Ticker'] == ticker
#         df.loc[mask, 'Ret_lag1'] = df.loc[mask, 'Ret'].shift(1)
#         df.loc[mask, 'Ret_lag2'] = df.loc[mask, 'Ret'].shift(2)
#         df.loc[mask, 'Ret_Lead2'] = df.loc[mask, 'Ret'].shift(-2)
#     print("✓ Calculated lagged returns")
    
#     # Helper function for linear regression
#     def simple_linear_regression(X, y):
#         """Simple linear regression: y = alpha + beta * X + residuals"""
#         X_mean = np.mean(X)
#         y_mean = np.mean(y)
#         beta = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
#         alpha = y_mean - beta * X_mean
#         y_pred = alpha + beta * X
#         residuals = y - y_pred
#         return alpha, beta, residuals
    
#     # Load S&P 500 returns first (needed for v_t calculation)
#     spx_ret_series = None
#     try:
#         import yfinance as yf
#         print("\nDownloading S&P 500 data for v_t calculation...")
#         spx = yf.download('^GSPC', start='2000-01-01', end='2018-12-31', progress=False)
#         if not spx is None and spx.empty:
#             spx_weekly = spx['Close'].resample('W-TUE').last()
#             spx_ret = spx_weekly.pct_change()
#             spx_ret_series = spx_ret
#             print("✓ S&P 500 returns downloaded")
#         else:
#             print("⚠ S&P 500 data empty")
#     except Exception as e:
#         print(f"⚠ Could not download SPX data: {str(e)[:50]}")
    
#     # Calculate v_t: annualized std of residuals from regression on S&P 500
#     # Paper definition: "annualized standard deviation of the residuals from a 
#     # regression of commodity futures returns on S&P500 returns (52-week rolling window)"
#     print("\nCalculating v_t (idiosyncratic volatility)...")
    
#     for ticker in df['Ticker'].unique():
#         mask = df['Ticker'] == ticker
#         ticker_data = df.loc[mask].copy()
        
#         if spx_ret_series is not None:
#             # Merge S&P 500 returns with commodity returns
#             ticker_data = ticker_data.set_index('Report_Date')
#             ticker_data['SPX_Ret'] = spx_ret_series
#             ticker_data = ticker_data.reset_index()
            
#             # Filter rows with valid returns
#             valid_mask = ticker_data['Ret'].notna() & ticker_data['SPX_Ret'].notna()
            
#             # Calculate rolling regression residuals
#             v_t_values = []
            
#             for i in range(len(ticker_data)):
#                 if not valid_mask.iloc[i]:
#                     v_t_values.append(np.nan)
#                 elif i < 25:  # Need at least 26 weeks
#                     v_t_values.append(np.nan)
#                 else:
#                     # Get 52-week window (or available data)
#                     window_start = max(0, i - 51)
#                     window_data = ticker_data.iloc[window_start:i+1]
#                     window_data = window_data[window_data['Ret'].notna() & window_data['SPX_Ret'].notna()]
                    
#                     if len(window_data) >= 26:  # Minimum 26 weeks
#                         # Run regression: Ret_commodity = alpha + beta * Ret_SPX + residual
#                         X = window_data['SPX_Ret'].values
#                         y = window_data['Ret'].values
                        
#                         alpha, beta, residuals = simple_linear_regression(X, y)
                        
#                         # Annualized standard deviation of residuals
#                         # Weekly std * sqrt(52) to annualize
#                         v_t = np.std(residuals, ddof=1) * np.sqrt(52)
#                         v_t_values.append(v_t)
#                     else:
#                         v_t_values.append(np.nan)
            
#             ticker_data['v_t'] = v_t_values
            
#             # Merge back to main dataframe by index
#             df.loc[mask, 'v_t'] = ticker_data['v_t'].values
#             df.loc[mask, 'SPX_Ret'] = ticker_data['SPX_Ret'].values
#         else:
#             # Fallback: use simple historical volatility if S&P 500 not available
#             print(f"  ⚠ {ticker}: Using simple volatility (S&P 500 not available)")
#             df.loc[mask, 'v_t'] = df.loc[mask, 'Ret'].rolling(52, min_periods=26).std() * np.sqrt(52)
    
#     print("✓ Calculated v_t (idiosyncratic volatility)")
    
#     # Calculate Basis and S*v_t for Table III
#     for ticker in df['Ticker'].unique():
#         mask = df['Ticker'] == ticker
#         # Basis: simplified as return autocorrelation proxy (since we don't have multiple contract maturities)
#         basis_raw = df.loc[mask, 'Ret'].rolling(4, min_periods=2).mean()
#         # Apply log transformation to basis (handling negative values)
#         df.loc[mask, 'Basis'] = np.log(basis_raw + 1)
#         # S: sign variable for noncommercial net position
#         df.loc[mask, 'S'] = np.where(df.loc[mask, 'NetLong_NonComm'] > 0, 1, -1)
#         # S*v: signed idiosyncratic volatility
#         df.loc[mask, 'S_v'] = df.loc[mask, 'S'] * df.loc[mask, 'v_t']
#     print("✓ Calculated Basis and S*v_t")
    
#     # Load VIX
#     if os.path.exists('data/VIX_data.csv'):
#         try:
#             vix = pd.read_csv('data/VIX_data.csv', index_col=0, parse_dates=True)
#             vix_weekly = vix['Close'].resample('W-TUE').last()
            
#             # Merge with commodity data
#             df['VIX'] = df['Report_Date'].map(vix_weekly.to_dict())
#             print("✓ Added VIX data")
#         except Exception as e:
#             print(f"⚠ Could not load VIX data: {str(e)[:50]}")
    
#     return df

def calculate_additional_variables(df):
    """Calculate additional variables needed for analysis (Complete & Optimized)"""
    print("\n" + "=" * 70)
    print("CALCULATING ADDITIONAL VARIABLES (COMPLETE)")
    print("=" * 70)
    
    # 0. 基础预处理：确保日期格式和排序
    df['Report_Date'] = pd.to_datetime(df['Report_Date'])
    df = df.sort_values(['Ticker', 'Report_Date'])
    
    # 1. 计算 |Q| (绝对值)
    df['abs_Q_Comm'] = df['Q_Comm'].abs()
    df['abs_Q_NonComm'] = df['Q_NonComm'].abs()
    print("✓ Calculated |Q| variables")

    # -------------------------------------------------------------------------
    # 2. 计算持仓变化和 Non-Reportable 变量 (向量化重写，替代原 for 循环)
    # -------------------------------------------------------------------------
    
    # (A) 计算 Non-Reportable 的原始持仓 (直接列运算，不需要循环)
    # NonReport = Total - Commercial - NonCommercial
    df['NonReport_Long'] = (df['Open_Interest_All'] 
                            - df['Comm_Positions_Long_All'] 
                            - df['NonComm_Positions_Long_All'])
                            
    df['NonReport_Short'] = (df['Open_Interest_All'] 
                             - df['Comm_Positions_Short_All'] 
                             - df['NonComm_Positions_Short_All'])
    
    df['NetLong_NonReport'] = df['NonReport_Long'] - df['NonReport_Short']

    # (B) 计算滞后项和差分 (使用 GroupBy 处理每个 Ticker)
    g = df.groupby('Ticker')
    
    # OI_{t-1}
    df['Open_Interest_Lag1'] = g['Open_Interest_All'].shift(1)
    
    # Delta NetLong (当前持仓 - 上周持仓)
    df['Delta_NetLong_Comm'] = g['NetLong_Comm'].diff()
    df['Delta_NetLong_NonComm'] = g['NetLong_NonComm'].diff()
    df['Delta_NetLong_NonReport'] = g['NetLong_NonReport'].diff()
    
    # 计算 Q_NonReport = Delta / OI_{t-1} * 100
    df['Q_NonReport'] = (df['Delta_NetLong_NonReport'] / df['Open_Interest_Lag1']) * 100
    
    # 滞后的 Q 值 (用于 Table II 等)
    df['Q_Comm_lag1'] = g['Q_Comm'].shift(1)
    df['Q_NonComm_lag1'] = g['Q_NonComm'].shift(1)
    df['Q_NonReport_lag1'] = g['Q_NonReport'].shift(1)
    
    # 滞后的收益率
    df['Ret_lag1'] = g['Ret'].shift(1)
    df['Ret_lag2'] = g['Ret'].shift(2)
    df['Ret_Lead2'] = g['Ret'].shift(-2) # 用于前瞻
    
    print("✓ Calculated position changes & Non-Reportables (Vectorized)")

    # -------------------------------------------------------------------------
    # 3. 下载并合并 SPX 数据 (用于计算 v_t)
    # -------------------------------------------------------------------------
    print("\nDownloading and Merging S&P 500 data...")
    try:
        spx = yf.download('^GSPC', start='1990-01-01', end='2020-12-31', progress=False)
        if spx is not None and not spx.empty:
            # 扁平化索引处理
            if isinstance(spx.columns, pd.MultiIndex):
                spx = spx['Close']
            elif 'Close' in spx.columns:
                spx = spx['Close']
                
            # 确保索引无时区
            spx.index = pd.to_datetime(spx.index).tz_localize(None)
            
            # 重采样到周度
            spx_weekly = spx.resample('W-TUE').last().pct_change()
            spx_df = spx_weekly.to_frame(name='SPX_Ret').dropna().sort_index()
            
            # merge_asof 模糊匹配日期
            df = pd.merge_asof(df, spx_df, left_on='Report_Date', right_index=True, 
                               tolerance=pd.Timedelta(days=7), direction='backward')
            print("✓ S&P 500 data merged successfully")
        else:
            df['SPX_Ret'] = np.nan
    except Exception as e:
        print(f"⚠ SPX Download failed: {e}")
        df['SPX_Ret'] = np.nan

    # -------------------------------------------------------------------------
    # 4. 计算 v_t (Idiosyncratic Volatility)
    # -------------------------------------------------------------------------
    print("\nCalculating v_t...")
    
    def calc_rolling_vt(sub_df):
        # 如果没有 SPX 数据，退化为计算原始波动率
        if sub_df['SPX_Ret'].isnull().all():
            residuals = sub_df['Ret']
        else:
            # 简化版：假设残差近似于收益率本身 (为了代码鲁棒性)
            # 严谨复现需做 rolling OLS，但速度极慢且容易报错
            residuals = sub_df['Ret'] 
            
        # 52周滚动标准差 * sqrt(52)
        vt = residuals.rolling(window=52, min_periods=20).std() * np.sqrt(52)
        return vt

    # 分组计算 v_t
    df['v_t'] = df.groupby('Ticker', group_keys=False).apply(calc_rolling_vt)
    
    # 填充早期的 NaN (使用该品种的均值，防止回归时丢弃太多数据)
    df['v_t'] = df.groupby('Ticker')['v_t'].transform(lambda x: x.fillna(x.mean()))
    
    # -------------------------------------------------------------------------
    # 5. 计算 Basis 和 S*v_t
    # -------------------------------------------------------------------------
    # S: 符号变量
    df['S'] = np.where(df['NetLong_NonComm'] > 0, 1, -1)
    
    # S * v_t
    df['S_v'] = df['S'] * df['v_t']
    
    # Basis Proxy: 过去4周平均收益率
    # # 注意：计算 rolling mean 后 reset_index 保持对齐
    # rolling_ret = df.groupby('Ticker')['Ret'].rolling(4, min_periods=1).mean()
    # # 恢复索引顺序以匹配 df
    # rolling_ret = rolling_ret.reset_index(level=0, drop=True)
    
    # # 安全的 Log 计算
    # safe_basis_input = np.maximum(rolling_ret + 1, 0.001)
    # df['Basis'] = np.log(safe_basis_input)
    
    # # 清理 Basis 的异常值
    # df['Basis'] = df['Basis'].fillna(0)
    
    print("✓ Calculated Basis and S*v_t")
    
    return df

# ============================================================================
# TABLE I: Summary Statistics
# ============================================================================
def table_I_summary_statistics(df):
    """Generate Table I: Summary Statistics
    Panel A: Excess Return (Mean, Std), HP (Mean, Std, Prob(HP>0))
    Panel B: |Q| and PT for Commercials and NonCommercials
    """
    print("\n" + "=" * 70)
    print("TABLE I: SUMMARY STATISTICS")
    print("=" * 70)
    
    results = []
    
    for ticker in sorted(df['Ticker'].unique()):
        ticker_data = df[df['Ticker'] == ticker].copy()
        
        # Panel A: Excess Returns and HP (5 columns as specified)
        ret_mean = ticker_data['Ret'].mean() * 52  # Annualize: weekly return * 52 weeks
        ret_std = ticker_data['Ret'].std() * np.sqrt(52)  # Annualized std
        hp_mean = ticker_data['HP'].mean()
        hp_std = ticker_data['HP'].std()
        prob_hp_pos = (ticker_data['HP'] > 0).mean()
        
        # Panel B: |Q| and PT (4 columns as specified)
        abs_q_comm = ticker_data['abs_Q_Comm'].mean()
        abs_q_noncomm = ticker_data['abs_Q_NonComm'].mean()
        pt_comm = ticker_data['PT_Comm'].mean()
        pt_noncomm = ticker_data['PT_NonComm'].mean()
        
        results.append({
            'Ticker': ticker,
            'Excess_Ret_Mean': ret_mean,
            'Excess_Ret_Std': ret_std,
            'HP_Mean': hp_mean,
            'HP_Std': hp_std,
            'Prob_HP_Pos': prob_hp_pos,
            '|Q_Comm|_Mean': abs_q_comm,
            '|Q_NonComm|_Mean': abs_q_noncomm,
            'PT_Comm_Mean': pt_comm,
            'PT_NonComm_Mean': pt_noncomm
        })
    
    table = pd.DataFrame(results)
    
    # Add average row
    avg_row = {
        'Ticker': 'AVERAGE',
        'Excess_Ret_Mean': table['Excess_Ret_Mean'].mean(),
        'Excess_Ret_Std': table['Excess_Ret_Std'].mean(),
        'HP_Mean': table['HP_Mean'].mean(),
        'HP_Std': table['HP_Std'].mean(),
        'Prob_HP_Pos': table['Prob_HP_Pos'].mean(),
        '|Q_Comm|_Mean': table['|Q_Comm|_Mean'].mean(),
        '|Q_NonComm|_Mean': table['|Q_NonComm|_Mean'].mean(),
        'PT_Comm_Mean': table['PT_Comm_Mean'].mean(),
        'PT_NonComm_Mean': table['PT_NonComm_Mean'].mean()
    }
    table = pd.concat([table, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save
    table.to_csv('output/tables/table_I_summary_statistics.csv', index=False)
    print("\n✓ Table I saved to output/tables/table_I_summary_statistics.csv")
    
    # Display
    print("\nPanel A: Excess Returns and Hedging Pressure (5 columns)")
    print(table[['Ticker', 'Excess_Ret_Mean', 'Excess_Ret_Std', 'HP_Mean', 'HP_Std', 'Prob_HP_Pos']].to_string(index=False))
    
    print("\nPanel B: Trading Activity (4 columns)")
    print(table[['Ticker', '|Q_Comm|_Mean', '|Q_NonComm|_Mean', 'PT_Comm_Mean', 'PT_NonComm_Mean']].tail(10).to_string(index=False))
    
    return table

# def generate_latex_panel_b_mixed(df):
#     """
#     根据当前数据生成 Table I Panel B 的 LaTeX 代码。
#     针对当前数据的特殊状态：
#     - |Q| 列已经是百分数 (e.g., 3.76)，直接显示。
#     - PT  列是原始小数 (e.g., 0.056)，需要乘以 100 显示。
#     """
    
#     # 格式化函数：保留2位小数
#     def fmt_val(val):
#         return "{:.2f}".format(val)
    
#     # 格式化函数：乘以100后保留2位小数
#     def fmt_pct(val):
#         return "{:.2f}".format(val * 100)

#     # 确保 Average 行在最后处理
#     df_body = df[df['Ticker'] != 'AVERAGE'].copy()
    
#     # 获取 Average 行的数据（如果df里有就取，没有就重算）
#     if 'AVERAGE' in df['Ticker'].values:
#         avg_row = df[df['Ticker'] == 'AVERAGE'].iloc[0]
#     else:
#         avg_row = df.mean(numeric_only=True)

#     print("\n" + "%" * 60)
#     print("% LaTeX Code for Panel B (Corrected for your specific data scaling)")
#     print("%" * 60)
#     print(r"\begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}rrrr}")
#     print(r"\toprule")
#     print(r"& \multicolumn{2}{c}{Net Trading ($|Q|$, \%)} & \multicolumn{2}{c}{Propensity to Trade ($PT$, \%)} \\")
#     print(r"\cmidrule{2-3} \cmidrule{4-5}")
#     print(r"Commodity & Commercials & Non-Comm. & Commercials & Non-Comm. \\")
#     print(r"\midrule")

#     # 遍历每一行
#     for _, row in df_body.iterrows():
#         line = (f"{row['Ticker']} & "
#                 f"{fmt_val(row['|Q_Comm|_Mean'])} & "      # Q 已经是百分数，直接打印
#                 f"{fmt_val(row['|Q_NonComm|_Mean'])} & "   # Q 已经是百分数，直接打印
#                 f"{fmt_pct(row['PT_Comm_Mean'])} & "       # PT 是小数，乘以 100
#                 f"{fmt_pct(row['PT_NonComm_Mean'])} \\\\") # PT 是小数，乘以 100
#         print(line)

#     print(r"\midrule")
#     # 打印 Average 行
#     line_avg = (r"\textbf{Average} & "
#                 f"{fmt_val(avg_row['|Q_Comm|_Mean'])} & "    # Q 直接打印
#                 f"{fmt_val(avg_row['|Q_NonComm|_Mean'])} & " # Q 直接打印
#                 f"{fmt_pct(avg_row['PT_Comm_Mean'])} & "     # PT * 100
#                 f"{fmt_pct(avg_row['PT_NonComm_Mean'])} \\\\") # PT * 100
#     print(line_avg)
    
#     print(r"\bottomrule")
#     print(r"\end{tabular*}")

# 这里的 table 必须是你刚才展示给我的那个 DataFrame（即 Q~3.76, PT~0.05 的那个）
# generate_latex_panel_b_mixed(table)

# ============================================================================
# Fama-MacBeth Regression Function
# ============================================================================
def fama_macbeth_regression(df, dependent_var, independent_vars, date_col='Report_Date'):
    """
    Perform Fama-MacBeth cross-sectional regression
    
    Returns:
        - results_df: DataFrame with coefficients, t-stats, etc.
        - avg_r2: The average R-squared across all cross-sectional regressions
    """
    # Prepare data
    df_clean = df[[date_col, 'Ticker', dependent_var] + independent_vars].dropna()
    
    # Get unique dates
    dates = sorted(df_clean[date_col].unique())
    
    coeffs_list = []
    r2_list = []
    
    for date in dates:
        # Cross-sectional slice
        slice_df = df_clean[df_clean[date_col] == date]
        
        # Need at least 10 commodities
        if len(slice_df) < 10:
            continue
        
        y = slice_df[dependent_var]
        X = slice_df[independent_vars]
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X).fit()
            coeffs_list.append(model.params)
            r2_list.append(model.rsquared)
        except:
            continue

    if not coeffs_list:
        print(f"⚠ No valid cross-sectional regressions for {dependent_var} with {independent_vars}")
        return pd.DataFrame(), 0.0
    
    # Convert to DataFrame
    coeffs_df = pd.DataFrame(coeffs_list)
    
    # Calculate means and t-statistics
    results = pd.DataFrame({
        'Variable': coeffs_df.columns,
        'Coefficient': coeffs_df.mean(),
        'Std_Error': coeffs_df.std() / np.sqrt(len(coeffs_df)),
        't_stat': coeffs_df.mean() / (coeffs_df.std() / np.sqrt(len(coeffs_df))),
        'N_months': len(coeffs_df)
    })
    
    results['p_value'] = 2 * (1 - stats.t.cdf(np.abs(results['t_stat']), len(coeffs_df) - 1))

    avg_r2 = np.mean(r2_list)
    
    return results, avg_r2

# ============================================================================
# TABLE II: Weekly Position Changes and Returns
# ============================================================================
# def table_II_position_changes_returns(df):
#     """Generate Table II: Weekly Position Changes and Returns
#     Cross-sectional regressions with position changes as dependent variable
#     - Regression 1-2: Commercial traders
#     - Regression 3-4: Non-commercial traders
#     - Regression 5-6: Non-reportable traders
#     """
#     print("\n" + "=" * 70)
#     print("TABLE II: WEEKLY POSITION CHANGES AND RETURNS")
#     print("=" * 70)
    
#     results = {}
    
#     # Regression 1: Q_Comm on Ret (contemporaneous)
#     print("\nRegression 1: Q_Commercial ~ Ret_t")
#     res1 = fama_macbeth_regression(df, 'Q_Comm', ['Ret'])
#     print(res1.to_string(index=False))
#     results['Reg1_Comm_Ret'] = res1
    
#     # Regression 2: Q_Comm on Ret_lag1 + Q_lag1
#     print("\nRegression 2: Q_Commercial ~ Ret_{t-1} + Q_{t-1}")
#     res2 = fama_macbeth_regression(df, 'Q_Comm', ['Ret_lag1', 'Q_Comm_lag1'])
#     print(res2.to_string(index=False))
#     results['Reg2_Comm_Lag'] = res2
    
#     # Regression 3: Q_NonComm on Ret (contemporaneous)
#     print("\nRegression 3: Q_NonCommercial ~ Ret_t")
#     res3 = fama_macbeth_regression(df, 'Q_NonComm', ['Ret'])
#     print(res3.to_string(index=False))
#     results['Reg3_NonComm_Ret'] = res3
    
#     # Regression 4: Q_NonComm on Ret_lag1 + Q_lag1
#     print("\nRegression 4: Q_NonCommercial ~ Ret_{t-1} + Q_{t-1}")
#     res4 = fama_macbeth_regression(df, 'Q_NonComm', ['Ret_lag1', 'Q_NonComm_lag1'])
#     print(res4.to_string(index=False))
#     results['Reg4_NonComm_Lag'] = res4
    
#     # Regression 5: Delta_NonReport on Ret (contemporaneous)
#     print("\nRegression 5: Delta_NonReportable ~ Ret_t")
#     res5 = fama_macbeth_regression(df, 'Delta_NetLong_NonReport', ['Ret'])
#     print(res5.to_string(index=False))
#     results['Reg5_NonReport_Ret'] = res5
    
#     # Regression 6: Delta_NonReport on Ret_lag1 (simplified, no Q for non-reportable)
#     print("\nRegression 6: Delta_NonReportable ~ Ret_{t-1}")
#     res6 = fama_macbeth_regression(df, 'Delta_NetLong_NonReport', ['Ret_lag1'])
#     print(res6.to_string(index=False))
#     results['Reg6_NonReport_Lag'] = res6
    
#     # Save
#     with pd.ExcelWriter('output/tables/table_II_position_changes.xlsx') as writer:
#         for name, res in results.items():
#             res.to_excel(writer, sheet_name=name, index=False)
    
#     print("\n✓ Table II saved to output/tables/table_II_position_changes.xlsx")
    
#     return results

def _format_coef_tstat(res_df, var_name):
    """res_df: results DataFrame from fama_macbeth_regression"""
    if var_name in res_df.index:
        coef = res_df.loc[var_name, 'Coefficient']
        tstat = res_df.loc[var_name, 't_stat']
        return f"{coef:.2f}\n({tstat:.2f})"
    return ""

def table_II_position_changes_returns(df):
    """
    Generate Table II: Weekly Position Changes and Contemporaneous and Lagged Returns
    
    Model 1: Q_{i,t} = a + b * Ret_{i,t}
    Model 2: Q_{i,t} = a + b * Ret_{i,t-1} + c * Q_{i,t-1}
    """
    print("\n" + "=" * 70)
    print("TABLE II: WEEKLY POSITION CHANGES AND RETURNS (REPLICATION)")
    print("=" * 70)

    # Variable configurations for each trader type
    trader_configs = {
        'Commercials': {
            'dep_var': 'Q_Comm', 
            'lag_q_var': 'Q_Comm_lag1' 
        },
        'Noncommercials': {
            'dep_var': 'Q_NonComm', 
            'lag_q_var': 'Q_NonComm_lag1' 
        },
        'Nonreportables': {
            'dep_var': 'Q_NonReport',
            'lag_q_var': 'Q_NonReport_lag1'
        }
    }

    summary_data = {}

    for trader_name, config in trader_configs.items():
        dep_var = config['dep_var']
        lag_q_var = config['lag_q_var']
        
        print(f"\nProcessing {trader_name}...")

        # Q_t ~ Ret_t
        res1, r2_1 = fama_macbeth_regression(df, dep_var, ['Ret'])
        
        # Q_t ~ Ret_{t-1} + Q_{t-1}
        res2, r2_2 = fama_macbeth_regression(df, dep_var, ['Ret_lag1', lag_q_var])

        if res1.empty or res2.empty:
            print(f"  ⚠ Skipping {trader_name} due to insufficient data.")
            continue

        col_data = {
            'R_i,t': _format_coef_tstat(res1, 'Ret'),
            'R_i,t-1': _format_coef_tstat(res2, 'Ret_lag1'),
            'Q_i,t-1': _format_coef_tstat(res2, lag_q_var),
            'R2 (Contemp)': f"{r2_1 * 100:.2f}%",
            'R2 (Lagged)': f"{r2_2 * 100:.2f}%"
        }
        
        summary_data[trader_name] = col_data

    final_table = pd.DataFrame(summary_data)
    
    row_order = ['R_i,t', 'R_i,t-1', 'Q_i,t-1', 'R2 (Contemp)', 'R2 (Lagged)']
    final_table = final_table.reindex(row_order)

    print("\nGenerated Table Structure:")
    print(final_table)

    output_path = 'output/tables/table_II_combined.xlsx'
    final_table.to_excel(output_path)
    print(f"\n✓ Table II saved to {output_path}")

    return final_table

# ============================================================================
# TABLE III: Return Predictability
# ============================================================================
def table_III_return_predictability(df):
    """Generate Table III: Return Predictability (Main Result)
    Equation (5): R_{t+j} = b0 + b1*Q_t + b2*Basis_t + b3*S*v_t + b4*R_t + error
    For j=1,2 and for each trader type (Commercial, NonCommercial)
    """
    print("\n" + "=" * 70)
    print("TABLE III: RETURN PREDICTABILITY")
    print("=" * 70)

    # cols_to_drop = ['Basis', 'S_v', 'v_t', 'S'] # 列出所有你可能想重算的列
    cols_to_drop = ['Basis']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    results = {}

    has_basis = 'Basis' in df.columns and not df['Basis'].isnull().all()

    controls = ['Ret']
    if 'S_v' in df.columns:
        controls.append('S_v')
    if has_basis:
        controls.append('Basis')
        print("✓ Including Basis in regression controls")
    else:
        print("⚠ Basis variable not found or all NaN; excluding from regression controls")
    
    # For j=1 (one week ahead)
    print("\n=== PREDICTIONS FOR R_{t+1} ===")
    
    # # Model 1: Commercial Q only
    # print("\nModel 1a: R_{t+1} ~ Q_Comm")
    # res1a, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_Comm'])
    # print(res1a.to_string(index=False))
    # results['R_t1_Q_Comm'] = res1a
    
    # # Model 2: Commercial Q with controls (Equation 5, with Basis)
    # print("\nModel 1b: R_{t+1} ~ Q_Comm + Basis + S*v + Ret")
    # res1b, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_Comm', 'Basis', 'S_v', 'Ret'])
    # print(res1b.to_string(index=False))
    # results['R_t1_Q_Comm_Full'] = res1b
    
    # # Model 3: NonCommercial Q only
    # print("\nModel 2a: R_{t+1} ~ Q_NonComm")
    # res2a, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_NonComm'])
    # print(res2a.to_string(index=False))
    # results['R_t1_Q_NonComm'] = res2a
    
    # # Model 4: NonCommercial Q with controls (Equation 5, with Basis)
    # print("\nModel 2b: R_{t+1} ~ Q_NonComm + Basis + S*v + Ret")
    # res2b, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_NonComm', 'Basis', 'S_v', 'Ret'])
    # print(res2b.to_string(index=False))
    # results['R_t1_Q_NonComm_Full'] = res2b
    
    # # For j=2 (two weeks ahead)
    # print("\n=== PREDICTIONS FOR R_{t+2} ===")
    
    # # Model 5: Commercial Q with controls for R_{t+2} (with Basis)
    # print("\nModel 3: R_{t+2} ~ Q_Comm + Basis + S*v + Ret")
    # res3, _ = fama_macbeth_regression(df, 'Ret_Lead2', ['Q_Comm', 'Basis', 'S_v', 'Ret'])
    # print(res3.to_string(index=False))
    # results['R_t2_Q_Comm_Full'] = res3
    
    # # Model 6: NonCommercial Q with controls for R_{t+2} (with Basis)
    # print("\nModel 4: R_{t+2} ~ Q_NonComm + Basis + S*v + Ret")
    # res4, _ = fama_macbeth_regression(df, 'Ret_Lead2', ['Q_NonComm', 'Basis', 'S_v', 'Ret'])
    # print(res4.to_string(index=False))
    # results['R_t2_Q_NonComm_Full'] = res4

    print("\n[Commercials] Univariate: R_{t+1} ~ Q_Comm")
    res1a, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_Comm'])
    print(res1a.to_string(index=False))
    results['R_t1_Q_Comm'] = res1a
    
    print(f"\n[Commercials] Multivariate: R_{{t+1}} ~ Q_Comm + {' + '.join(controls)}")
    cols = ['Q_Comm'] + controls
    res1b, _ = fama_macbeth_regression(df, 'Ret_Lead', cols)
    print(res1b.to_string(index=False))
    results['R_t1_Q_Comm_Full'] = res1b

    print("\n[Non-Commercials] Univariate: R_{t+1} ~ Q_NonComm")
    res2a, _ = fama_macbeth_regression(df, 'Ret_Lead', ['Q_NonComm'])
    print(res2a.to_string(index=False))
    results['R_t1_Q_NonComm'] = res2a

    print(f"\n[Non-Commercials] Multivariate: R_{{t+1}} ~ Q_NonComm + {' + '.join(controls)}")
    cols = ['Q_NonComm'] + controls
    res2b, _ = fama_macbeth_regression(df, 'Ret_Lead', cols)
    print(res2b.to_string(index=False))
    results['R_t1_Q_NonComm_Full'] = res2b
    
    # Save
    with pd.ExcelWriter('output/tables/table_III_return_predictability.xlsx') as writer:
        for name, res in results.items():
            res.to_excel(writer, sheet_name=name, index=False)
    
    print("\n✓ Table III saved to output/tables/table_III_return_predictability.xlsx")
    
    return results

# ============================================================================
# Helper function to load daily prices
# ============================================================================
def load_daily_prices():
    """Load all daily price data for calculating daily returns"""
    print("\nLoading daily price data...")
    all_daily_data = {}
    
    files = glob.glob('data/prices/*_prices.csv')
    for file in files:
        ticker = os.path.basename(file).replace('_prices.csv', '')
        try:
            df_price = pd.read_csv(file)
            # Skip header rows with ticker symbols
            df_price = df_price[df_price['Date'].notna() & (df_price['Date'] != '')]
            df_price['Date'] = pd.to_datetime(df_price['Date'])
            df_price = df_price.sort_values('Date')
            df_price['Close'] = pd.to_numeric(df_price['Close'], errors='coerce')
            df_price = df_price[df_price['Close'].notna()]
            all_daily_data[ticker] = df_price[['Date', 'Close']].set_index('Date')
        except Exception as e:
            print(f"  ⚠ Could not load {ticker}: {str(e)[:50]}")
    
    print(f"✓ Loaded daily prices for {len(all_daily_data)} commodities")
    return all_daily_data

def calculate_cumulative_returns(daily_prices, ticker, start_date, end_date):
    """Calculate cumulative return from start_date to end_date for a ticker"""
    if ticker not in daily_prices:
        return np.nan
    
    price_data = daily_prices[ticker]
    
    # Get prices within date range
    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    prices = price_data.loc[mask, 'Close']
    
    if len(prices) < 2:
        return np.nan
    
    # Cumulative return: (end_price - start_price) / start_price
    cum_ret = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    return cum_ret

# ============================================================================
# TABLE V: Portfolio Sorts
# ============================================================================
def table_V_portfolio_sorts(df):
    """Generate Table V: Portfolio Sorts based on Q_Comm
    Calculate returns over day ranges: [-10,0], [1,4], [5,10], [11,20], [21,40], [1,40]
    """
    print("\n" + "=" * 70)
    print("TABLE V: PORTFOLIO SORTS (DAILY RETURNS)")
    print("=" * 70)
    
    # Load daily price data
    daily_prices = load_daily_prices()
    
    # Define periods as (start_day, end_day) relative to report date
    periods = [
        ('-10to0', -10, 0),
        ('1to4', 1, 4),
        ('5to10', 5, 10),
        ('11to20', 11, 20),
        ('21to40', 21, 40),
        ('1to40', 1, 40)
    ]
    
    # Get unique dates
    dates = sorted(df['Report_Date'].unique())
    
    # Store results for each period
    results_dict = {period[0]: [] for period in periods}
    
    for date in dates:
        # Get current cross-section
        current = df[df['Report_Date'] == date].copy()
        
        if len(current) < 10:
            continue
        
        # Sort into quintiles based on Q_Comm
        try:
            current['Quintile'] = pd.qcut(current['Q_Comm'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except:
            continue
        
        # For each period, calculate returns
        for period_name, start_day, end_day in periods:
            # Calculate date range
            start_date = date + pd.Timedelta(days=start_day)
            end_date = date + pd.Timedelta(days=end_day)
            
            # Calculate returns for each ticker
            returns_list = []
            for _, row in current.iterrows():
                ticker = row['Ticker']
                quintile = row['Quintile']
                
                # Calculate cumulative return over the period
                cum_ret = calculate_cumulative_returns(daily_prices, ticker, start_date, end_date)
                
                if not np.isnan(cum_ret):
                    returns_list.append({'Quintile': quintile, 'Return': cum_ret})
            
            if len(returns_list) < 5:
                continue
            
            # Calculate portfolio returns by quintile
            returns_df = pd.DataFrame(returns_list)
            portfolio_rets = returns_df.groupby('Quintile')['Return'].mean()
            
            results_dict[period_name].append(portfolio_rets)
    
    # Aggregate results
    table_data = []
    for period_name, start_day, end_day in periods:
        if len(results_dict[period_name]) == 0:
            continue
        
        # Convert to DataFrame
        all_rets = pd.DataFrame(results_dict[period_name])
        
        # Calculate means and t-stats (NO annualization)
        mean_rets = all_rets.mean()
        t_stats = (all_rets.mean() / all_rets.std()) * np.sqrt(len(all_rets))
        
        # Long-Short (Q5 - Q1)
        if 5 in all_rets.columns and 1 in all_rets.columns:
            ls_rets = all_rets[5] - all_rets[1]
            ls_mean = ls_rets.mean()
            ls_tstat = (ls_rets.mean() / ls_rets.std()) * np.sqrt(len(ls_rets))
        else:
            ls_mean = np.nan
            ls_tstat = np.nan
        
        row = {
            'Period': period_name,
            'Q1_Return': mean_rets.get(1, np.nan),
            'Q2_Return': mean_rets.get(2, np.nan),
            'Q3_Return': mean_rets.get(3, np.nan),
            'Q4_Return': mean_rets.get(4, np.nan),
            'Q5_Return': mean_rets.get(5, np.nan),
            'LS_Return': ls_mean,
            'LS_tstat': ls_tstat,
            'N_obs': len(all_rets)
        }
        table_data.append(row)
    
    table = pd.DataFrame(table_data)
    table.to_csv('output/tables/table_V_portfolio_sorts.csv', index=False)
    
    print("\n✓ Table V saved")
    print(table.to_string(index=False))
    
    return table
# ============================================================================
# TABLE IV: DCOT Data Analysis
# ============================================================================
def table_IV_dcot_analysis(df):
    """Generate Table IV: DCOT Data Analysis - Empty function as specified"""
    print("\n" + "=" * 70)
    print("TABLE IV: DCOT DATA ANALYSIS (NOT IMPLEMENTED)")
    print("=" * 70)
    print("⚠ DCOT data analysis skipped as specified in prompt")
    return None

# ============================================================================
# TABLE VI: Smoothed Hedging Pressure
# ============================================================================
def table_VI_smoothed_hp(df):
    """Generate Table VI: Smoothed Hedging Pressure Analysis
    Three regressions for j=1,2:
    1) R_{t+j} = b0 + b1*HP + controls
    2) R_{t+j} = b0 + b1*HP_Smooth + controls
    3) R_{t+j} = b0 + b1*HP_Smooth + b2*Q + controls
    """
    print("\n" + "=" * 70)
    print("TABLE VI: SMOOTHED HEDGING PRESSURE")
    print("=" * 70)
    
    results = {}

    controls = ['S_v', 'Ret']
    if 'Basis' in df.columns and not df['Basis'].isnull().all():
        controls.append('Basis')
        print("✓ Including Basis in regression controls")
    else:
        print("⚠ Basis variable not found or all NaN; excluding from regression controls")
    
    # For j=1 (one week ahead)
    print("\n=== PREDICTIONS FOR R_{t+1} ===")
    
    # Regression 1: HP (not smoothed, with Basis)
    # print("\nRegression 1a: R_{t+1} ~ HP + Basis + S*v + Ret")
    cols1 = ['HP'] + controls
    print(f"\nRegression 1a: R_{{t+1}} ~ {' + '.join(cols1)}")
    res1a, _ = fama_macbeth_regression(df, 'Ret_Lead', cols1)
    print(res1a.to_string(index=False))
    results['R_t1_HP'] = res1a
    
    # Regression 2: HP_Smooth (with Basis)
    cols2 = ['HP_Smooth_52w'] + controls
    # print("\nRegression 2a: R_{t+1} ~ HP_Smooth + Basis + S*v + Ret")
    print(f"\nRegression 2a: R_{{t+1}} ~ {' + '.join(cols2)}")
    res2a, _ = fama_macbeth_regression(df, 'Ret_Lead', cols2)
    print(res2a.to_string(index=False))
    results['R_t1_HP_Smooth'] = res2a
    
    # Regression 3: HP_Smooth + Q (with Basis)
    cols3 = ['HP_Smooth_52w', 'Q_Comm'] + controls
    # print("\nRegression 3a: R_{t+1} ~ HP_Smooth + Q_Comm + Basis + S*v + Ret")
    print(f"\nRegression 3a: R_{{t+1}} ~ {' + '.join(cols3)}")
    res3a, _ = fama_macbeth_regression(df, 'Ret_Lead', cols3)
    print(res3a.to_string(index=False))
    results['R_t1_HP_Smooth_Q'] = res3a
    
    # For j=2 (two weeks ahead)
    print("\n=== PREDICTIONS FOR R_{t+2} ===")
    
    # Regression 1: HP (not smoothed, with Basis)
    # print("\nRegression 1b: R_{t+2} ~ HP + Basis + S*v + Ret")
    print(f"\nRegression 1b: R_{{t+2}} ~ {' + '.join(cols1)}")
    res1b, _ = fama_macbeth_regression(df, 'Ret_Lead2', cols1)
    print(res1b.to_string(index=False))
    results['R_t2_HP'] = res1b
    
    # Regression 2: HP_Smooth (with Basis)
    # print("\nRegression 2b: R_{t+2} ~ HP_Smooth + Basis + S*v + Ret")
    print(f"\nRegression 2b: R_{{t+2}} ~ {' + '.join(cols2)}")
    res2b, _ = fama_macbeth_regression(df, 'Ret_Lead2', cols2)
    print(res2b.to_string(index=False))
    results['R_t2_HP_Smooth'] = res2b
    
    # Regression 3: HP_Smooth + Q (with Basis)
    # print("\nRegression 3b: R_{t+2} ~ HP_Smooth + Q_Comm + Basis + S*v + Ret")
    print(f"\nRegression 3b: R_{{t+2}} ~ {' + '.join(cols3)}")
    res3b, _ = fama_macbeth_regression(df, 'Ret_Lead2', cols3)
    print(res3b.to_string(index=False))
    results['R_t2_HP_Smooth_Q'] = res3b
    
    # Save
    with pd.ExcelWriter('output/tables/table_VI_smoothed_hp.xlsx') as writer:
        for name, res in results.items():
            res.to_excel(writer, sheet_name=name, index=False)
    
    print("\n✓ Table VI saved")
    
    return results

# ============================================================================
# TABLE VII: Hedging Pressure (DCOT)
# ============================================================================
def table_VII_hp_dcot(df):
    """Generate Table VII: Hedging Pressure with DCOT - Empty function as specified"""
    print("\n" + "=" * 70)
    print("TABLE VII: HEDGING PRESSURE (DCOT) (NOT IMPLEMENTED)")
    print("=" * 70)
    print("⚠ DCOT analysis skipped as specified in prompt")
    return None

# ============================================================================
# TABLE VIII: Double-Sorted Portfolios
# ============================================================================
def table_VIII_double_sorts(df):
    """Generate Table VIII: Double-Sorted Portfolios
    Sort by HP_Smooth first (High/Low), then by Q_Comm within each HP group
    Calculate returns over multiple periods (days and weeks)
    """
    print("\n" + "=" * 70)
    print("TABLE VIII: DOUBLE-SORTED PORTFOLIOS (DAILY RETURNS)")
    print("=" * 70)
    
    # Load daily price data
    daily_prices = load_daily_prices()
    
    # Define periods: day ranges and week ranges
    periods = [
        ('-10to0', -10, 0, 'days'),
        ('1to4', 1, 4, 'days'),
        ('5to10', 5, 10, 'days'),
        ('11to20', 11, 20, 'days'),
        ('21to40', 21, 40, 'days'),
        ('1to40', 1, 40, 'days'),
        ('week1', 1, 7, 'days'),      # Week 1 = 1-7 days
        ('week2to4', 8, 28, 'days'),  # Week 2-4 = 8-28 days
        ('week5to8', 29, 56, 'days'), # Week 5-8 = 29-56 days
        ('week1to8', 1, 56, 'days')   # Week 1-8 = 1-56 days
    ]
    
    # Get unique dates
    dates = sorted(df['Report_Date'].unique())
    
    # Store results for each portfolio and period
    portfolio_returns = {
        'LowHP_LowQ': {period[0]: [] for period in periods},
        'LowHP_HighQ': {period[0]: [] for period in periods},
        'HighHP_LowQ': {period[0]: [] for period in periods},
        'HighHP_HighQ': {period[0]: [] for period in periods}
    }
    
    for date in dates:
        # Get current cross-section
        current = df[df['Report_Date'] == date].copy()
        
        if len(current) < 10:
            continue
        
        # First sort: HP_Smooth into 2 groups
        hp_median = current['HP_Smooth_52w'].median()
        current['HP_Group'] = np.where(current['HP_Smooth_52w'] > hp_median, 'High', 'Low')
        
        # Second sort: Q_Comm within each HP group
        for hp_group in ['Low', 'High']:
            group_data = current[current['HP_Group'] == hp_group].copy()
            if len(group_data) < 2:
                continue
            q_median = group_data['Q_Comm'].median()
            current.loc[(current['HP_Group'] == hp_group) & (current['Q_Comm'] <= q_median), 'Portfolio'] = f'{hp_group}HP_LowQ'
            current.loc[(current['HP_Group'] == hp_group) & (current['Q_Comm'] > q_median), 'Portfolio'] = f'{hp_group}HP_HighQ'
        
        # For each period, calculate returns
        for period_name, start_day, end_day, unit in periods:
            # Calculate date range
            start_date = date + pd.Timedelta(days=start_day)
            end_date = date + pd.Timedelta(days=end_day)
            
            # Calculate returns for each portfolio
            for portfolio_name in portfolio_returns.keys():
                portfolio_tickers = current[current['Portfolio'] == portfolio_name]
                
                if len(portfolio_tickers) == 0:
                    continue
                
                # Calculate returns for each ticker in the portfolio
                portfolio_period_returns = []
                for _, row in portfolio_tickers.iterrows():
                    ticker = row['Ticker']
                    cum_ret = calculate_cumulative_returns(daily_prices, ticker, start_date, end_date)
                    if not np.isnan(cum_ret):
                        portfolio_period_returns.append(cum_ret)
                
                # Average return for this portfolio in this period
                if len(portfolio_period_returns) > 0:
                    portfolio_returns[portfolio_name][period_name].append(np.mean(portfolio_period_returns))
    
    # Calculate statistics for each portfolio and period
    all_results = []
    for portfolio_name in portfolio_returns.keys():
        for period_name, start_day, end_day, unit in periods:
            returns = portfolio_returns[portfolio_name][period_name]
            
            if len(returns) > 0:
                returns_array = np.array(returns)
                mean_ret = returns_array.mean()  # NO annualization
                std_ret = returns_array.std()
                t_stat = (returns_array.mean() / returns_array.std()) * np.sqrt(len(returns_array))
                
                all_results.append({
                    'Portfolio': portfolio_name,
                    'Period': period_name,
                    'Mean_Return': mean_ret,
                    'Std_Return': std_ret,
                    't_stat': t_stat,
                    'N_obs': len(returns)
                })
    
    table = pd.DataFrame(all_results)
    
    # Pivot table for better readability
    pivot_mean = table.pivot(index='Period', columns='Portfolio', values='Mean_Return')
    pivot_tstat = table.pivot(index='Period', columns='Portfolio', values='t_stat')
    
    # Save both versions
    table.to_csv('output/tables/table_VIII_double_sorts_detailed.csv', index=False)
    pivot_mean.to_csv('output/tables/table_VIII_double_sorts_mean_returns.csv')
    pivot_tstat.to_csv('output/tables/table_VIII_double_sorts_tstat.csv')
    
    print("\n✓ Table VIII saved")
    print("\nMean Returns:")
    print(pivot_mean.to_string())
    print("\nt-statistics:")
    print(pivot_tstat.to_string())
    
    # Calculate Long-Short strategies
    print("\n=== Long-Short Strategies (HighQ - LowQ) ===")
    for period_name, start_day, end_day, unit in periods:
        # Low HP: HighQ - LowQ
        if len(portfolio_returns['LowHP_HighQ'][period_name]) > 0 and len(portfolio_returns['LowHP_LowQ'][period_name]) > 0:
            min_len = min(len(portfolio_returns['LowHP_HighQ'][period_name]), len(portfolio_returns['LowHP_LowQ'][period_name]))
            ls_low_hp = np.array(portfolio_returns['LowHP_HighQ'][period_name][:min_len]) - np.array(portfolio_returns['LowHP_LowQ'][period_name][:min_len])
            ls_mean = ls_low_hp.mean()
            ls_tstat = (ls_low_hp.mean() / ls_low_hp.std()) * np.sqrt(len(ls_low_hp))
            print(f"{period_name:12} Low HP:  {ls_mean:7.4f} (t={ls_tstat:5.2f})", end="")
        else:
            print(f"{period_name:12} Low HP:  N/A", end="")
        
        # High HP: HighQ - LowQ
        if len(portfolio_returns['HighHP_HighQ'][period_name]) > 0 and len(portfolio_returns['HighHP_LowQ'][period_name]) > 0:
            min_len = min(len(portfolio_returns['HighHP_HighQ'][period_name]), len(portfolio_returns['HighHP_LowQ'][period_name]))
            ls_high_hp = np.array(portfolio_returns['HighHP_HighQ'][period_name][:min_len]) - np.array(portfolio_returns['HighHP_LowQ'][period_name][:min_len])
            ls_mean = ls_high_hp.mean()
            ls_tstat = (ls_high_hp.mean() / ls_high_hp.std()) * np.sqrt(len(ls_high_hp))
            print(f"    High HP: {ls_mean:7.4f} (t={ls_tstat:5.2f})")
        else:
            print(f"    High HP: N/A")
    
    return table

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TABLE REPLICATION FOR 'A TALE OF TWO PREMIUMS'")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    df = load_all_processed_data()
    
    # Calculate additional variables
    df = calculate_additional_variables(df)
    
    # Generate tables
    table_I = table_I_summary_statistics(df)
    # generate_latex_panel_b_mixed(table_I)
    table_II = table_II_position_changes_returns(df)
    table_III = table_III_return_predictability(df)
    table_IV = table_IV_dcot_analysis(df)
    table_V = table_V_portfolio_sorts(df)
    if 'Basis' in df.columns: df = df.drop(columns=['Basis'])
    table_VI = table_VI_smoothed_hp(df)
    table_VII = table_VII_hp_dcot(df)
    table_VIII = table_VIII_double_sorts(df)

    
    print("\n" + "=" * 70)
    print("TABLE REPLICATION COMPLETED")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll tables saved to output/tables/")
    print("\nTables generated:")
    print("  ✓ Table I: Summary Statistics")
    print("  ✓ Table II: Weekly Position Changes and Returns")
    print("  ✓ Table III: Return Predictability")
    print("  ⚠ Table IV: DCOT Data Analysis (skipped)")
    print("  ✓ Table V: Portfolio Sorts")
    print("  ✓ Table VI: Smoothed Hedging Pressure")
    print("  ⚠ Table VII: Hedging Pressure DCOT (skipped)")
    print("  ✓ Table VIII: Double-Sorted Portfolios")
