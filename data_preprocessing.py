"""
Data Preprocessing Script for "A Tale of Two Premiums" Paper Replication
Aligns time series, constructs variables according to paper formulas
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob

def load_cftc_data():
    """
    Load and preprocess CFTC data
    """
    print("=" * 60)
    print("Loading CFTC Data...")
    print("=" * 60)
    
    # Load Legacy COT data
    legacy_file = 'data/cftc_legacy/legacy_cot_data.csv'
    if os.path.exists(legacy_file):
        print(f"Loading Legacy COT data... ", end='')
        legacy_df = pd.read_csv(legacy_file, low_memory=False)
        print(f"✓ ({len(legacy_df)} records)")
    else:
        print(f"✗ Legacy COT data not found")
        legacy_df = None
    
    # Load Disaggregated COT data
    disagg_file = 'data/cftc_disagg/disagg_cot_data.csv'
    if os.path.exists(disagg_file):
        print(f"Loading Disaggregated COT data... ", end='')
        disagg_df = pd.read_csv(disagg_file, low_memory=False)
        print(f"✓ ({len(disagg_df)} records)")
    else:
        print(f"✗ Disaggregated COT data not found")
        disagg_df = None
    
    return legacy_df, disagg_df

def process_cftc_legacy(df):
    """
    Process Legacy COT data with robust cleaning
    """
    if df is None:
        return None
    
    print("\nProcessing Legacy COT data...")
    
    df.columns = df.columns.str.strip()
    
    date_col = None
    if 'As of Date in Form YYMMDD' in df.columns:
        date_col = 'As of Date in Form YYMMDD'
        df[date_col] = df[date_col].astype(str).str.zfill(6)
        df['Report_Date'] = pd.to_datetime(df[date_col], format='%y%m%d', errors='coerce')
    elif 'Report_Date_as_MM_DD_YYYY' in df.columns:
        df['Report_Date'] = pd.to_datetime(df['Report_Date_as_MM_DD_YYYY'], errors='coerce')
    else:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df['Report_Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    
    if 'Report_Date' not in df.columns:
        print("✗ Could not parse Report Date")
        return None

    required_cols = {
        'Open_Interest_All': ['Open Interest (All)', 'Open_Interest_All', 'OI_All'],
        'NonComm_Positions_Long_All': ['Noncommercial Positions-Long (All)', 'NonComm_Positions_Long_All', 'Noncommercial Long'],
        'NonComm_Positions_Short_All': ['Noncommercial Positions-Short (All)', 'NonComm_Positions_Short_All', 'Noncommercial Short'],
        'Comm_Positions_Long_All': ['Commercial Positions-Long (All)', 'Comm_Positions_Long_All', 'Commercial Long'],
        'Comm_Positions_Short_All': ['Commercial Positions-Short (All)', 'Comm_Positions_Short_All', 'Commercial Short'],
        'CFTC_Contract_Market_Code': ['CFTC Contract Market Code', 'CFTC_Contract_Market_Code', 'CFTC Code'],
        'Market_and_Exchange_Names': ['Market and Exchange Names', 'Market_and_Exchange_Names', 'Market']
    }
    
    column_map = {}
    for target_col, possible_names in required_cols.items():
        for col in df.columns:
            if col in possible_names:
                column_map[col] = target_col
                break
    
    df_processed = df.rename(columns=column_map)
    
    # Select columns
    keep_cols = ['Report_Date'] + list(required_cols.keys())
    keep_cols = [col for col in keep_cols if col in df_processed.columns]
    df_processed = df_processed[keep_cols].copy()
    
    if 'CFTC_Contract_Market_Code' in df_processed.columns:
        df_processed['CFTC_Contract_Market_Code'] = df_processed['CFTC_Contract_Market_Code'].apply(
            lambda x: str(int(float(x))).strip().zfill(6) if pd.notnull(x) and str(x).replace('.','').isdigit() else str(x).strip()
        )

    numeric_cols = ['Open_Interest_All', 'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
                    'Comm_Positions_Long_All', 'Comm_Positions_Short_All']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    df_processed = df_processed.dropna(subset=['Report_Date', 'Open_Interest_All'])
    df_processed = df_processed.sort_values('Report_Date')
    
    print(f"✓ Processed {len(df_processed)} records (Date Range: {df_processed['Report_Date'].min().date()} to {df_processed['Report_Date'].max().date()})")

    return df_processed

def process_cftc_disaggregated(df):
    """
    Process Disaggregated COT data
    """
    if df is None:
        return None
    
    print("\nProcessing Disaggregated COT data...")
    
    # Parse date column
    if 'Report_Date_as_MM_DD_YYYY' in df.columns:
        df['Report_Date'] = pd.to_datetime(df['Report_Date_as_MM_DD_YYYY'], errors='coerce')
    elif 'As_of_Date_In_Form_YYMMDD' in df.columns:
        df['Report_Date'] = pd.to_datetime(df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d', errors='coerce')
    else:
        print("✗ Could not find report date column")
        return None
    
    # Extract relevant columns for disaggregated data
    required_cols = {
        'Open_Interest_All': ['Open_Interest_All', 'OI_All'],
        'Prod_Merc_Positions_Long_All': ['Prod_Merc_Positions_Long_All'],
        'Prod_Merc_Positions_Short_All': ['Prod_Merc_Positions_Short_All'],
        'Swap_Positions_Long_All': ['Swap_Positions_Long_All'],
        'Swap_Positions_Short_All': ['Swap__Positions_Short_All', 'Swap_Positions_Short_All'],
        'M_Money_Positions_Long_All': ['M_Money_Positions_Long_All'],
        'M_Money_Positions_Short_All': ['M_Money_Positions_Short_All'],
        'CFTC_Contract_Market_Code': ['CFTC_Contract_Market_Code', 'CFTC_Market_Code'],
        'Market_and_Exchange_Names': ['Market_and_Exchange_Names']
    }
    
    # Map columns
    column_map = {}
    for target_col, possible_names in required_cols.items():
        for col in df.columns:
            if col in possible_names:
                column_map[col] = target_col
                break
    
    df_processed = df.rename(columns=column_map)
    
    # Select columns
    keep_cols = ['Report_Date'] + list(required_cols.keys())
    keep_cols = [col for col in keep_cols if col in df_processed.columns]
    df_processed = df_processed[keep_cols].copy()
    
    # Convert numeric columns
    numeric_cols = [col for col in required_cols.keys() if 'Position' in col or 'Interest' in col]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Remove rows with missing critical data
    df_processed = df_processed.dropna(subset=['Report_Date', 'Open_Interest_All'])
    
    print(f"✓ Processed {len(df_processed)} records")
    return df_processed

def load_and_resample_prices():
    """
    Load commodity price data and resample to weekly (Tuesday)
    Includes dirty data cleaning
    """
    print("\n" + "=" * 60)
    print("Loading and Resampling Price Data...")
    print("=" * 60)
    
    price_files = glob.glob('data/prices/*_prices.csv')
    price_dict = {}
    
    for file in price_files:
        ticker = os.path.basename(file).replace('_prices.csv', '')
        try:
            # Read price file
            df = pd.read_csv(file)
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            if 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            df = df.dropna(subset=['Date', 'Close'])
            
            df = df.set_index('Date')
            df = df.sort_index()
            
            # Resample to weekly, using Tuesday as anchor
            if not df.empty:
                df_weekly = df.resample('W-TUE').last()
                
                df_weekly = df_weekly.dropna(subset=['Close'])
                
                if not df_weekly.empty:
                    price_dict[ticker] = df_weekly[['Close']].rename(columns={'Close': f'{ticker}_Close'})
                    print(f"✓ {ticker:12} - {len(df_weekly)} weekly observations ({df_weekly.index.min().date()} to {df_weekly.index.max().date()})")
                else:
                    print(f"✗ {ticker:12} - Empty after resampling")
            else:
                print(f"✗ {ticker:12} - Empty after cleaning")
            
        except Exception as e:
            print(f"✗ {ticker:12} - Error: {str(e)[:40]}")
            continue
    
    return price_dict

def calculate_variables(df):
    """
    Calculate variables according to paper equations (1)-(4)
    
    Parameters:
    -----------
    df : DataFrame with columns:
        - Open_Interest_All (OI)
        - Comm_Positions_Long_All, Comm_Positions_Short_All
        - NonComm_Positions_Long_All, NonComm_Positions_Short_All
        - Close price
    
    Returns:
    --------
    df : DataFrame with additional columns:
        - HP: Hedging Pressure
        - Q_Comm, Q_NonComm: Net Trading
        - Ret: Excess Return
    """
    print("\n" + "=" * 60)
    print("Calculating Variables...")
    print("=" * 60)
    
    # Equation (1): Hedging Pressure (HP)
    # HP = (Comm_Short - Comm_Long) / OI
    if 'Comm_Positions_Short_All' in df.columns and 'Comm_Positions_Long_All' in df.columns:
        df['HP'] = (df['Comm_Positions_Short_All'] - df['Comm_Positions_Long_All']) / df['Open_Interest_All']
        print("✓ Calculated Hedging Pressure (HP)")
    
    # Equation (2): Net Trading (Q)
    # Q = (NetLong_t - NetLong_{t-1}) / OI_{t-1} * 100  (in percentage)
    # NOTE: Do NOT use absolute value - Q can be positive or negative
    # Positive Q = increasing net long position
    # Negative Q = decreasing net long position
    
    # For Commercial (Hedgers)
    if 'Comm_Positions_Long_All' in df.columns and 'Comm_Positions_Short_All' in df.columns:
        df['NetLong_Comm'] = df['Comm_Positions_Long_All'] - df['Comm_Positions_Short_All']
        df['Q_Comm'] = df['NetLong_Comm'].diff() / df['Open_Interest_All'].shift(1) * 100
        print("✓ Calculated Net Trading for Commercial (Q_Comm) in percentage (without abs)")
    
    # For Non-Commercial (Speculators)
    if 'NonComm_Positions_Long_All' in df.columns and 'NonComm_Positions_Short_All' in df.columns:
        df['NetLong_NonComm'] = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
        df['Q_NonComm'] = df['NetLong_NonComm'].diff() / df['Open_Interest_All'].shift(1) * 100
        print("✓ Calculated Net Trading for Non-Commercial (Q_NonComm) in percentage (without abs)")
    
    # Equation (3): Propensity to Trade (PT)
    # PT = (|ΔLong| + |ΔShort|) / (Long + Short)
    
    if 'Comm_Positions_Long_All' in df.columns and 'Comm_Positions_Short_All' in df.columns:
        delta_long_comm = df['Comm_Positions_Long_All'].diff().abs()
        delta_short_comm = df['Comm_Positions_Short_All'].diff().abs()
        total_comm = df['Comm_Positions_Long_All'] + df['Comm_Positions_Short_All']
        df['PT_Comm'] = (delta_long_comm + delta_short_comm) / total_comm
        print("✓ Calculated Propensity to Trade for Commercial (PT_Comm)")
    
    if 'NonComm_Positions_Long_All' in df.columns and 'NonComm_Positions_Short_All' in df.columns:
        delta_long_noncomm = df['NonComm_Positions_Long_All'].diff().abs()
        delta_short_noncomm = df['NonComm_Positions_Short_All'].diff().abs()
        total_noncomm = df['NonComm_Positions_Long_All'] + df['NonComm_Positions_Short_All']
        df['PT_NonComm'] = (delta_long_noncomm + delta_short_noncomm) / total_noncomm
        print("✓ Calculated Propensity to Trade for Non-Commercial (PT_NonComm)")
    
    # Equation (4): Excess Return (R)
    # R_{t+1} = (F_{t+1} - F_t) / F_t
    price_col = [col for col in df.columns if 'Close' in col]
    if price_col:
        price_col = price_col[0]
        df['Ret'] = df[price_col].pct_change()
        # Lead return by 1 period (next week's return)
        df['Ret_Lead'] = df['Ret'].shift(-1)
        print(f"✓ Calculated Returns using {price_col}")
    
    # Smoothed HP: 52-week rolling average
    if 'HP' in df.columns:
        df['HP_Smooth_52w'] = df['HP'].rolling(window=52, min_periods=26).mean()
        print("✓ Calculated 52-week smoothed HP")
    
    return df

def merge_cot_and_prices(cot_df, price_dict, commodity_map):
    """
    Merge COT data with price data based on commodity matching
    
    Parameters:
    -----------
    cot_df : DataFrame with COT data
    price_dict : dict of {ticker: price_df}
    commodity_map : tuple of (name_map, code_map)
    
    Returns:
    --------
    merged_dict : dict of {ticker: merged_df}
    """
    print("\n" + "=" * 60)
    print("Merging COT and Price Data...")
    print("=" * 60)
    
    name_map, code_map = commodity_map
    merged_dict = {}
    
    for ticker, price_df in price_dict.items():
        # Try code-based matching first (more precise)
        if ticker in code_map:
            cftc_code = code_map[ticker]
            cot_subset = cot_df[cot_df['CFTC_Contract_Market_Code'] == cftc_code]
            match_method = f"CFTC Code {cftc_code}"
        else:
            # Fall back to name-based matching
            commodity_name = name_map.get(ticker, ticker)
            cot_subset = cot_df[cot_df['Market_and_Exchange_Names'].str.contains(
                commodity_name, case=False, na=False)]
            match_method = f"Name '{commodity_name}'"
        
        if not cot_subset.empty:
            # Set Report_Date as index
            cot_subset = cot_subset.set_index('Report_Date')
            
            # Merge on index (dates)
            merged = cot_subset.join(price_df, how='inner')
            
            if not merged.empty:
                # Sort by date in ascending order
                merged = merged.sort_index()
                merged_dict[ticker] = merged
                print(f"✓ {ticker:12} - {len(merged):4} obs via {match_method}")
            else:
                print(f"✗ {ticker:12} - No overlapping dates")
        else:
            print(f"✗ {ticker:12} - Not found in COT data ({match_method})")
    
    return merged_dict

def create_commodity_map():
    """
    Create mapping between Yahoo Finance tickers and CFTC identifiers
    Returns two mappings: one for names (general) and one for CFTC codes (specific)
    """
    # Name-based mapping (for most commodities)
    name_map = {
        'CL': 'CRUDE OIL',
        'HO': 'HEATING OIL',
        'NG': 'NATURAL GAS',
        'RB': 'GASOLINE',
        'GC': 'GOLD',
        'SI': 'SILVER',
        'HG': 'COPPER',
        'PL': 'PLATINUM',
        'PA': 'PALLADIUM',
        'ZC': 'CORN',
        'ZO': 'OATS',
        'ZS': 'SOYBEANS',
        'ZL': 'SOYBEAN OIL',
        'ZM': 'SOYBEAN MEAL',
        'RR': 'RICE',
        'KC': 'COFFEE',
        'SB': 'SUGAR',
        'CC': 'COCOA',
        'CT': 'COTTON',
        'OJ': 'ORANGE JUICE',
        'LB': 'LUMBER',
        'LE': 'LIVE CATTLE',
        'HE': 'LEAN HOGS',
        'GF': 'FEEDER CATTLE'
    }
    
    # CFTC Code-based mapping for all commodities
    # Using explicit codes ensures accurate matching and avoids ambiguity
    code_map = {
        # Energy
        'CL': '067651',  # Crude Oil WTI - NYMEX (not ICE Europe)
        'HO': '022651',  # Heating Oil - NYMEX (main contract, not swaps)
        'NG': '023651',  # Natural Gas - NYMEX (not ICE)
        
        # Precious Metals
        'GC': '088691',  # Gold - COMEX (not CBOT)
        'SI': '084691',  # Silver - COMEX (not CBOT)
        'PL': '076651',  # Platinum - NYMEX
        'PA': '075651',  # Palladium - NYMEX
        
        # Base Metals
        'HG': '085692',  # Copper - COMEX
        
        # Grains 缺少 Minn Wheat
        'ZW': '001602',  # Wheat SRW - CBOT Chicago
        'KE': '001612',  # Wheat HRW - KCBT Kansas City
        'ZC': '002602',  # Corn - CBOT
        'ZO': '004603',  # Oats - CBOT
        'ZS': '005602',  # Soybeans - CBOT
        'ZL': '007601',  # Soybean Oil - CBOT
        'ZM': '026603',  # Soybean Meal - CBOT
        'RR': '039601',  # Rough Rice - CBOT
        
        # Softs
        'KC': '083731',  # Coffee - ICE (formerly CSCE)
        'SB': '080732',  # Sugar #11 - ICE (not #14)
        'CC': '073732',  # Cocoa - ICE
        'CT': '033661',  # Cotton #2 - ICE
        'OJ': '040701',  # Orange Juice - ICE
        
        # Livestock
        'LB': '058643',  # Lumber - CME
        'LE': '057642',  # Live Cattle - CME
        'HE': '054642',  # Lean Hogs - CME
        'GF': '061641',  # Feeder Cattle - CME
    }
    
    return name_map, code_map

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING FOR TWO PREMIUMS PAPER REPLICATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Load CFTC data
    legacy_df, disagg_df = load_cftc_data()
    
    # 2. Process CFTC data
    if legacy_df is not None:
        legacy_processed = process_cftc_legacy(legacy_df)
    else:
        legacy_processed = None
    
    if disagg_df is not None:
        disagg_processed = process_cftc_disaggregated(disagg_df)
    else:
        disagg_processed = None
    
    # 3. Load and resample prices
    price_dict = load_and_resample_prices()
    
    # 4. Create commodity mapping
    commodity_map = create_commodity_map()
    
    # 5. Merge data and calculate variables
    if legacy_processed is not None and price_dict:
        merged_dict = merge_cot_and_prices(legacy_processed, price_dict, commodity_map)
        
        # Calculate variables for each commodity
        os.makedirs('data/processed', exist_ok=True)
        
        for ticker, df in merged_dict.items():
            df_with_vars = calculate_variables(df)
            # Ensure data is sorted by date before saving
            df_with_vars = df_with_vars.sort_index()
            output_file = f'data/processed/{ticker}_processed.csv'
            df_with_vars.to_csv(output_file)
            print(f"\n✓ Saved processed data: {output_file}")
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nProcessed data saved in ./data/processed/ directory")
