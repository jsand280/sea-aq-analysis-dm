import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

plt.style.use('seaborn-v0_8-whitegrid')
CITY_COLORS = {
    'Bangkok': '#E74C3C',
    'Ho Chi Minh City': '#3498DB',
    'Kuala Lumpur': '#27AE60',
    'Singapore': '#F39C12'
}

def load_data():
    filepath = os.path.join(PROCESSED_DIR, 'combined_data.csv')
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    return df

def plot_pm25_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]['pm25']
        axes[0].hist(city_data, bins=50, alpha=0.6, label=city, color=CITY_COLORS.get(city))
    axes[0].set_xlabel('PM2.5 (µg/m³)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PM2.5 Distribution by City')
    axes[0].legend()
    axes[0].set_xlim(0, 150)
    
    city_order = df.groupby('city')['pm25'].median().sort_values(ascending=False).index
    box_data = [df[df['city'] == city]['pm25'] for city in city_order]
    bp = axes[1].boxplot(box_data, labels=city_order, patch_artist=True)
    for patch, city in zip(bp['boxes'], city_order):
        patch.set_facecolor(CITY_COLORS.get(city, '#888888'))
        patch.set_alpha(0.7)
    axes[1].set_ylabel('PM2.5 (µg/m³)')
    axes[1].set_title('PM2.5 Box Plot by City')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pm25_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pm25_distribution.png")

def plot_time_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    daily = df.groupby(['city', df['datetime'].dt.date])['pm25'].mean().reset_index()
    daily['datetime'] = pd.to_datetime(daily['datetime'])
    
    for city in df['city'].unique():
        city_data = daily[daily['city'] == city]
        axes[0].plot(city_data['datetime'], city_data['pm25'], 
                     label=city, color=CITY_COLORS.get(city), alpha=0.8, linewidth=1)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('PM2.5 (µg/m³)')
    axes[0].set_title('Daily Average PM2.5 Over Time')
    axes[0].legend(loc='upper right')
    axes[0].axhline(y=35.4, color='red', linestyle='--', alpha=0.5, label='Unhealthy threshold')
    
    monthly = df.groupby(['city', 'month'])['pm25'].mean().reset_index()
    for city in df['city'].unique():
        city_data = monthly[monthly['city'] == city]
        axes[1].plot(city_data['month'], city_data['pm25'], 
                     marker='o', label=city, color=CITY_COLORS.get(city), linewidth=2)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('PM2.5 (µg/m³)')
    axes[1].set_title('Monthly Average PM2.5 (Seasonal Pattern)')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[1].legend(loc='upper right')
    axes[1].axhline(y=35.4, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'time_series.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: time_series.png")

def plot_hourly_pattern(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hourly = df.groupby(['city', 'hour'])['pm25'].mean().reset_index()
    for city in df['city'].unique():
        city_data = hourly[hourly['city'] == city]
        axes[0].plot(city_data['hour'], city_data['pm25'], 
                     marker='o', label=city, color=CITY_COLORS.get(city), linewidth=2)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('PM2.5 (µg/m³)')
    axes[0].set_title('Hourly PM2.5 Pattern')
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend()
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = df.groupby(['city', 'day_of_week'])['pm25'].mean().reset_index()
    for city in df['city'].unique():
        city_data = daily[daily['city'] == city]
        axes[1].plot(city_data['day_of_week'], city_data['pm25'], 
                     marker='o', label=city, color=CITY_COLORS.get(city), linewidth=2)
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('PM2.5 (µg/m³)')
    axes[1].set_title('Day of Week PM2.5 Pattern')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_names)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: temporal_patterns.png")

def plot_hazard_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hazard_order = ['good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'hazardous']
    hazard_colors = ['#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#8E44AD']
    
    hazard_counts = df.groupby(['city', 'hazard_level']).size().unstack(fill_value=0)
    hazard_pct = hazard_counts.div(hazard_counts.sum(axis=1), axis=0) * 100
    hazard_pct = hazard_pct.reindex(columns=hazard_order)
    
    hazard_pct.plot(kind='bar', stacked=True, ax=axes[0], color=hazard_colors, edgecolor='white')
    axes[0].set_xlabel('City')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Air Quality Distribution by City')
    axes[0].legend(title='Hazard Level', bbox_to_anchor=(1.02, 1))
    axes[0].tick_params(axis='x', rotation=15)
    
    season_hazard = df.groupby(['season', 'hazard_level']).size().unstack(fill_value=0)
    season_hazard_pct = season_hazard.div(season_hazard.sum(axis=1), axis=0) * 100
    season_order = ['dry', 'transition', 'wet']
    season_hazard_pct = season_hazard_pct.reindex(index=season_order, columns=hazard_order)
    
    season_hazard_pct.plot(kind='bar', stacked=True, ax=axes[1], color=hazard_colors, edgecolor='white')
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Air Quality Distribution by Season')
    axes[1].legend(title='Hazard Level', bbox_to_anchor=(1.02, 1))
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hazard_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hazard_distribution.png")

def plot_correlation_heatmap(df):
    numeric_cols = ['pm25', 'hour', 'day_of_week', 'month', 'is_weekend',
                    'pm25_lag_1h', 'pm25_lag_24h', 'pm25_rolling_mean_24h']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', square=True, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")

def plot_city_comparison(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats = df.groupby('city')['pm25'].agg(['mean', 'std', 'median']).reset_index()
    stats = stats.sort_values('mean', ascending=True)
    colors = [CITY_COLORS.get(city) for city in stats['city']]
    axes[0, 0].barh(stats['city'], stats['mean'], xerr=stats['std'], color=colors, alpha=0.8, capsize=5)
    axes[0, 0].set_xlabel('PM2.5 (µg/m³)')
    axes[0, 0].set_title('Mean PM2.5 by City (with Std Dev)')
    axes[0, 0].axvline(x=35.4, color='red', linestyle='--', alpha=0.7)
    
    unhealthy_pct = df.groupby('city')['is_unhealthy'].mean() * 100
    unhealthy_pct = unhealthy_pct.sort_values(ascending=True)
    colors = [CITY_COLORS.get(city) for city in unhealthy_pct.index]
    axes[0, 1].barh(unhealthy_pct.index, unhealthy_pct.values, color=colors, alpha=0.8)
    axes[0, 1].set_xlabel('Percentage (%)')
    axes[0, 1].set_title('Percentage of Unhealthy Hours by City')
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]['pm25']
        axes[1, 0].hist(city_data, bins=50, alpha=0.5, label=city, 
                        color=CITY_COLORS.get(city), density=True)
    axes[1, 0].set_xlabel('PM2.5 (µg/m³)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('PM2.5 Density Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 120)
    
    season_city = df.groupby(['city', 'season'])['pm25'].mean().unstack()
    season_city = season_city[['dry', 'transition', 'wet']]
    season_city.plot(kind='bar', ax=axes[1, 1], color=['#E74C3C', '#F39C12', '#3498DB'], edgecolor='white')
    axes[1, 1].set_xlabel('City')
    axes[1, 1].set_ylabel('PM2.5 (µg/m³)')
    axes[1, 1].set_title('Seasonal PM2.5 by City')
    axes[1, 1].legend(title='Season')
    axes[1, 1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'city_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: city_comparison.png")

def generate_eda_report(df):
    report = []
    report.append("=" * 60)
    report.append("EXPLORATORY DATA ANALYSIS REPORT")
    report.append("Southeast Asia Air Quality Project")
    report.append("=" * 60)
    
    report.append("\n1. DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total records: {len(df):,}")
    report.append(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    report.append(f"Cities: {', '.join(df['city'].unique())}")
    report.append(f"Features: {len(df.columns)}")
    
    report.append("\n2. PM2.5 STATISTICS BY CITY")
    report.append("-" * 40)
    stats = df.groupby('city')['pm25'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
    report.append(stats.round(2).to_string())
    
    report.append("\n3. AIR QUALITY CLASSIFICATION")
    report.append("-" * 40)
    hazard_pct = df.groupby('city')['hazard_level'].value_counts(normalize=True).unstack() * 100
    report.append(hazard_pct.round(1).to_string())
    
    report.append("\n4. TEMPORAL PATTERNS")
    report.append("-" * 40)
    report.append("\nPeak pollution hours (highest mean PM2.5):")
    peak_hours = df.groupby('hour')['pm25'].mean().nlargest(3)
    for hour, pm25 in peak_hours.items():
        report.append(f"  Hour {hour:02d}:00 - {pm25:.1f} µg/m³")
    
    report.append("\nSeasonal comparison:")
    seasonal = df.groupby('season')['pm25'].mean()
    for season, pm25 in seasonal.items():
        report.append(f"  {season.capitalize()}: {pm25:.1f} µg/m³")
    
    report.append("\n5. KEY FINDINGS")
    report.append("-" * 40)
    
    most_polluted = df.groupby('city')['pm25'].mean().idxmax()
    cleanest = df.groupby('city')['pm25'].mean().idxmin()
    report.append(f"- Most polluted city: {most_polluted}")
    report.append(f"- Cleanest city: {cleanest}")
    
    worst_season = df.groupby('season')['pm25'].mean().idxmax()
    report.append(f"- Worst season: {worst_season}")
    
    unhealthy_pct = (df['is_unhealthy'].mean() * 100)
    report.append(f"- Overall unhealthy hours: {unhealthy_pct:.1f}%")
    
    report_text = '\n'.join(report)
    
    report_file = os.path.join(FIGURES_DIR, 'eda_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nSaved: eda_report.txt")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading processed data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nGenerating visualizations...")
    plot_pm25_distribution(df)
    plot_time_series(df)
    plot_hourly_pattern(df)
    plot_hazard_distribution(df)
    plot_correlation_heatmap(df)
    plot_city_comparison(df)
    
    print("\nGenerating EDA report...")
    generate_eda_report(df)
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")

if __name__ == '__main__':
    main()
