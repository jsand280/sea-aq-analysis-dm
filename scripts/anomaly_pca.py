import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

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

def prepare_features(df):
    feature_cols = ['pm25', 'hour', 'day_of_week', 'month', 'pm25_rolling_mean_24h']
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, scaler

def detect_anomalies_zscore(df, threshold=3):
    """Statistical approach: Z-score based detection"""
    z_scores = np.abs(stats.zscore(df['pm25'].fillna(df['pm25'].median())))
    return z_scores > threshold

def detect_anomalies_lof(X_scaled, n_neighbors=20, contamination=0.05):
    """Density-based approach: Local Outlier Factor"""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(X_scaled)
    return predictions == -1

def detect_anomalies_ocsvm(X_scaled, nu=0.05):
    """One-class model approach: One-Class SVM"""
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    predictions = ocsvm.fit_predict(X_scaled)
    return predictions == -1

def perform_pca(X_scaled, n_components=None):
    if n_components is None:
        n_components = X_scaled.shape[1]
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return pca, X_pca

def plot_anomaly_comparison(df, anomaly_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(anomaly_dict.keys())
    colors = ['#E74C3C', '#3498DB', '#27AE60']
    
    counts = [anomaly_dict[m].sum() for m in methods]
    axes[0, 0].bar(methods, counts, color=colors)
    axes[0, 0].set_ylabel('Number of Anomalies')
    axes[0, 0].set_title('Anomaly Count by Method')
    for i, c in enumerate(counts):
        axes[0, 0].text(i, c + 50, str(c), ha='center')
    
    monthly = df.groupby('month').apply(
        lambda x: pd.Series({m: anomaly_dict[m][x.index].mean()*100 for m in methods})
    )
    monthly.plot(kind='line', ax=axes[0, 1], marker='o')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Anomaly Rate (%)')
    axes[0, 1].set_title('Monthly Anomaly Rate by Method')
    axes[0, 1].legend(title='Method')
    axes[0, 1].grid(True, alpha=0.3)
    
    hourly = df.groupby('hour').apply(
        lambda x: pd.Series({m: anomaly_dict[m][x.index].mean()*100 for m in methods})
    )
    hourly.plot(kind='line', ax=axes[1, 0], marker='o')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Anomaly Rate (%)')
    axes[1, 0].set_title('Hourly Anomaly Rate by Method')
    axes[1, 0].legend(title='Method')
    axes[1, 0].grid(True, alpha=0.3)
    
    n_methods = sum([anomaly_dict[m].astype(int) for m in methods])
    agreement_counts = pd.Series(n_methods).value_counts().sort_index()
    axes[1, 1].bar(agreement_counts.index, agreement_counts.values, color='steelblue')
    axes[1, 1].set_xlabel('Number of Methods Agreeing')
    axes[1, 1].set_ylabel('Number of Observations')
    axes[1, 1].set_title('Method Agreement Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'anomaly_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: anomaly_comparison.png")

def plot_anomaly_distribution(df, anomaly_consensus):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_plot = df.copy()
    df_plot['anomaly'] = anomaly_consensus
    
    axes[0].hist(df_plot[~df_plot['anomaly']]['pm25'], bins=50, alpha=0.7, 
                 label='Normal', color='steelblue', density=True)
    axes[0].hist(df_plot[df_plot['anomaly']]['pm25'], bins=30, alpha=0.7, 
                 label='Anomaly', color='red', density=True)
    axes[0].set_xlabel('PM2.5 (µg/m³)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('PM2.5 Distribution: Normal vs Anomaly')
    axes[0].legend()
    
    city_rates = df_plot.groupby('city')['anomaly'].mean() * 100
    colors = [CITY_COLORS[city] for city in city_rates.index]
    bars = axes[1].bar(city_rates.index, city_rates.values, color=colors)
    axes[1].set_ylabel('Anomaly Rate (%)')
    axes[1].set_title('Consensus Anomaly Rate by City')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, city_rates.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'anomaly_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: anomaly_distribution.png")

def plot_anomalies_by_city(df, anomaly_consensus):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    cities = df['city'].unique()
    
    for idx, city in enumerate(cities):
        city_data = df[df['city'] == city].copy()
        city_data['anomaly'] = anomaly_consensus[df['city'] == city]
        
        daily = city_data.groupby(city_data['datetime'].dt.date).agg({
            'pm25': 'mean',
            'anomaly': 'sum'
        })
        
        ax = axes[idx]
        ax.plot(daily.index, daily['pm25'], color=CITY_COLORS[city], alpha=0.7, linewidth=0.8)
        
        anomaly_days = daily[daily['anomaly'] > 0]
        ax.scatter(anomaly_days.index, anomaly_days['pm25'], 
                  color='red', s=20, alpha=0.6, label='Anomaly days', zorder=5)
        
        ax.set_title(f'{city}')
        ax.set_xlabel('Date')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Anomaly Detection Results by City (Consensus: >=2 methods)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'anomalies_by_city.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: anomalies_by_city.png")

def plot_pca_results(pca, X_pca, df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, alpha=0.7, label='Individual')
    axes[0, 0].plot(range(1, len(cumsum) + 1), cumsum, 'ro-', label='Cumulative')
    axes[0, 0].axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('PCA: Explained Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for city in df['city'].unique():
        mask = df['city'] == city
        axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=CITY_COLORS[city], label=city, alpha=0.3, s=5)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 1].set_title('PCA: City Distribution')
    axes[0, 1].legend(markerscale=3)
    
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=df['pm25'], cmap='RdYlGn_r', alpha=0.3, s=5)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 0].set_title('PCA: PM2.5 Gradient')
    plt.colorbar(scatter, ax=axes[1, 0], label='PM2.5 (µg/m³)')
    
    feature_cols = ['pm25', 'hour', 'day_of_week', 'month', 'rolling_mean']
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_cols
    )
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('PCA: Component Loadings')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pca_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pca_analysis.png")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("="*60)
    print("ANOMALY DETECTION & PCA ANALYSIS")
    print("Methods: Z-score (statistical), LOF (density), One-Class SVM")
    print("="*60)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Total records: {len(df)}")
    
    if 'is_extreme' in df.columns:
        print(f"Pre-flagged extreme values: {df['is_extreme'].sum()}")
    
    print("\nPreparing features...")
    X_scaled, feature_cols, scaler = prepare_features(df)
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION (3 In-Scope Methods)")
    print("="*60)
    
    print("\n1. Z-score (Statistical approach)...")
    anomaly_zscore = detect_anomalies_zscore(df, threshold=3)
    print(f"   Anomalies: {anomaly_zscore.sum()} ({100*anomaly_zscore.mean():.2f}%)")
    
    print("\n2. Local Outlier Factor (Density-based approach)...")
    anomaly_lof = detect_anomalies_lof(X_scaled, contamination=0.05)
    print(f"   Anomalies: {anomaly_lof.sum()} ({100*anomaly_lof.mean():.2f}%)")
    
    print("\n3. One-Class SVM (One-class model approach)...")
    anomaly_ocsvm = detect_anomalies_ocsvm(X_scaled, nu=0.05)
    print(f"   Anomalies: {anomaly_ocsvm.sum()} ({100*anomaly_ocsvm.mean():.2f}%)")
    
    anomaly_dict = {
        'Z-score': anomaly_zscore,
        'LOF': anomaly_lof,
        'One-Class SVM': anomaly_ocsvm
    }
    
    print("\n" + "="*60)
    print("CONSENSUS ANOMALIES (>=2 methods agreeing)")
    print("="*60)
    
    n_methods = anomaly_zscore.astype(int) + anomaly_lof.astype(int) + anomaly_ocsvm.astype(int)
    anomaly_consensus = n_methods >= 2
    
    print(f"\nConsensus anomalies: {anomaly_consensus.sum()} ({100*anomaly_consensus.mean():.2f}%)")
    
    print("\nAnomaly rate by city:")
    for city in df['city'].unique():
        city_rate = anomaly_consensus[df['city'] == city].mean() * 100
        print(f"  {city}: {city_rate:.2f}%")
    
    print("\nAnomaly rate by season:")
    for season in ['dry', 'transition', 'wet']:
        season_rate = anomaly_consensus[df['season'] == season].mean() * 100
        print(f"  {season}: {season_rate:.2f}%")
    
    print("\n" + "="*60)
    print("TOP ANOMALY EVENTS")
    print("="*60)
    
    df_anomalies = df[anomaly_consensus].copy()
    df_anomalies = df_anomalies.sort_values('pm25', ascending=False)
    
    print("\nTop 10 highest PM2.5 anomaly events:")
    for _, row in df_anomalies.head(10).iterrows():
        print(f"  {row['datetime']} | {row['city']}: {row['pm25']:.1f} µg/m³")
    
    print("\n" + "="*60)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("="*60)
    
    pca, X_pca = perform_pca(X_scaled)
    
    print("\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cumsum >= 0.9) + 1
    print(f"\nComponents for 90% variance: {n_90}")
    print(f"Components for 95% variance: {np.argmax(cumsum >= 0.95) + 1}")
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_anomaly_comparison(df, anomaly_dict)
    plot_anomaly_distribution(df, anomaly_consensus)
    plot_anomalies_by_city(df, anomaly_consensus)
    plot_pca_results(pca, X_pca, df)
    
    df['anomaly_zscore'] = anomaly_zscore
    df['anomaly_lof'] = anomaly_lof
    df['anomaly_ocsvm'] = anomaly_ocsvm
    df['anomaly_consensus'] = anomaly_consensus
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    output_file = os.path.join(PROCESSED_DIR, 'data_with_anomalies.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved data with anomaly flags to: {output_file}")
    
    return anomaly_dict, pca

if __name__ == '__main__':
    anomalies, pca = main()
