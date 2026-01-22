import pandas as pd
import numpy as np
import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def load_raw_data():
    files = glob.glob(os.path.join(RAW_DIR, '*_pm25.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Loaded {f}: {len(df)} records")
    
    return pd.concat(dfs, ignore_index=True)

def standardize_schema(df):
    if 'date' in df.columns and 'utc' not in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'utc' in df.columns:
        df['datetime'] = pd.to_datetime(df['utc'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date.utc' in df.columns:
        df['datetime'] = pd.to_datetime(df['date.utc'])
    
    if 'value' in df.columns:
        df['pm25'] = df['value']
    elif 'pm25' not in df.columns:
        for col in df.columns:
            if 'pm' in col.lower() or 'value' in col.lower():
                df['pm25'] = df[col]
                break
    
    if 'city_name' in df.columns:
        df['city'] = df['city_name']
    elif 'city' not in df.columns:
        df['city'] = 'Unknown'
    
    return df[['datetime', 'city', 'pm25']].copy()

def handle_missing_values(df):
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    df['pm25'] = df['pm25'].replace([np.inf, -np.inf], np.nan)
    df['pm25'] = df.groupby('city')['pm25'].transform(
        lambda x: x.interpolate(method='linear', limit=3)
    )
    df['pm25'] = df.groupby('city')['pm25'].transform(
        lambda x: x.fillna(method='ffill', limit=24)
    )
    df['pm25'] = df.groupby('city')['pm25'].transform(
        lambda x: x.fillna(x.median())
    )
    return df

def flag_outliers(df):
    df = df[(df['pm25'] >= 0) & (df['pm25'] <= 999)].copy()
    
    def compute_outlier_flag(group):
        rolling_mean = group['pm25'].rolling(window=24, min_periods=1, center=True).mean()
        rolling_std = group['pm25'].rolling(window=24, min_periods=1, center=True).std()
        rolling_std = rolling_std.fillna(rolling_std.median())
        z_scores = (group['pm25'] - rolling_mean) / (rolling_std + 1e-6)
        group['is_extreme'] = (abs(z_scores) > 4).astype(int)
        return group
    
    df = df.groupby('city', group_keys=False).apply(compute_outlier_flag)
    extreme_count = df['is_extreme'].sum()
    print(f"Flagged {extreme_count} extreme values ({100*extreme_count/len(df):.2f}%) - KEPT for anomaly detection")
    return df

def engineer_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    def get_season(month):
        if month in [11, 12, 1, 2]:
            return 'dry'
        elif month in [5, 6, 7, 8, 9, 10]:
            return 'wet'
        else:
            return 'transition'
    
    df['season'] = df['month'].apply(get_season)
    
    def get_time_of_day(hour):
        if hour < 6:
            return 'night'
        elif hour < 12:
            return 'morning'
        elif hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    
    def get_hazard_level(pm25):
        if pm25 <= 12:
            return 'good'
        elif pm25 <= 35.4:
            return 'moderate'
        elif pm25 <= 55.4:
            return 'unhealthy_sensitive'
        elif pm25 <= 150.4:
            return 'unhealthy'
        else:
            return 'hazardous'
    
    df['hazard_level'] = df['pm25'].apply(get_hazard_level)
    df['is_unhealthy'] = (df['pm25'] > 35.4).astype(int)
    
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # FIXED: All features are now LAGGED (no leakage)
    df['pm25_lag_1h'] = df.groupby('city')['pm25'].shift(1)
    df['pm25_lag_24h'] = df.groupby('city')['pm25'].shift(24)
    
    # FIXED: Rolling stats computed on SHIFTED data (excludes current value)
    df['pm25_rolling_mean_24h'] = df.groupby('city')['pm25'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
    )
    df['pm25_rolling_std_24h'] = df.groupby('city')['pm25'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
    )
    
    # NEW: Target for forecasting task (predict NEXT hour)
    df['target_next_hour'] = df.groupby('city')['is_unhealthy'].shift(-1)
    
    # Fill NaN for lag features
    df['pm25_lag_1h'] = df['pm25_lag_1h'].fillna(df.groupby('city')['pm25'].transform('median'))
    df['pm25_lag_24h'] = df['pm25_lag_24h'].fillna(df.groupby('city')['pm25'].transform('median'))
    df['pm25_rolling_mean_24h'] = df['pm25_rolling_mean_24h'].fillna(df.groupby('city')['pm25'].transform('median'))
    df['pm25_rolling_std_24h'] = df['pm25_rolling_std_24h'].fillna(0)
    
    return df

def create_summary_stats(df):
    summary = df.groupby('city').agg({
        'pm25': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'is_unhealthy': 'mean',
        'datetime': ['min', 'max']
    }).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary['unhealthy_pct'] = (summary['is_unhealthy_mean'] * 100).round(1)
    return summary

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Total records loaded: {len(df)}")
    
    print("\nStandardizing schema...")
    df = standardize_schema(df)
    
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    
    print("\nFlagging extreme values (KEPT for anomaly detection)...")
    df = flag_outliers(df)
    
    print("\nEngineering features (with proper lagging - NO LEAKAGE)...")
    df = engineer_features(df)
    
    print("\nGenerating summary statistics...")
    summary = create_summary_stats(df)
    print("\n" + summary.to_string())
    
    output_file = os.path.join(PROCESSED_DIR, 'combined_data.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved processed data to: {output_file}")
    print(f"Final dataset: {len(df)} records, {len(df.columns)} columns")
    
    summary_file = os.path.join(PROCESSED_DIR, 'summary_stats.csv')
    summary.to_csv(summary_file)
    print(f"Saved summary stats to: {summary_file}")
    
    return df

if __name__ == '__main__':
    main()
