import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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

def prepare_clustering_features(df):
    feature_cols = ['pm25', 'hour', 'day_of_week', 'month']
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, scaler

def sampled_silhouette_score(X_scaled, labels, sample_size=5000, random_state=42):
    """Compute silhouette on a sample for efficiency"""
    if len(X_scaled) <= sample_size:
        return silhouette_score(X_scaled, labels)
    
    np.random.seed(random_state)
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    return silhouette_score(X_scaled[indices], labels[indices])

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    sse = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        sil = sampled_silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil)
        print(f"  k={k}: SSE={kmeans.inertia_:.0f}, Silhouette={sil:.4f}")
    
    return list(k_range), sse, silhouette_scores

def train_kmeans(X_scaled, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def train_dbscan(X_scaled, eps=0.8, min_samples=50):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return dbscan, labels

def train_gmm(X_scaled, n_components=2):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    return gmm, labels, probs

def evaluate_clustering(X_scaled, labels, method_name):
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            'Method': method_name,
            'N_Clusters': len(np.unique(labels[labels != -1])),
            'Noise_Points': (labels == -1).sum(),
            'Silhouette': np.nan,
            'Calinski_Harabasz': np.nan,
            'Davies_Bouldin': np.nan
        }
    
    sil = sampled_silhouette_score(X_scaled[mask], labels[mask])
    
    return {
        'Method': method_name,
        'N_Clusters': len(np.unique(labels[labels != -1])),
        'Noise_Points': (labels == -1).sum(),
        'Silhouette': sil,
        'Calinski_Harabasz': calinski_harabasz_score(X_scaled[mask], labels[mask]),
        'Davies_Bouldin': davies_bouldin_score(X_scaled[mask], labels[mask])
    }

def plot_elbow_and_silhouette(k_range, sse, silhouette_scores):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Sum of Squared Errors (SSE)')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score (sampled)')
    axes[1].set_title('Silhouette Analysis for Optimal k')
    axes[1].grid(True, alpha=0.3)
    
    best_k = k_range[np.argmax(silhouette_scores)]
    axes[1].axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'clustering_elbow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: clustering_elbow.png")

def plot_cluster_visualization(X_scaled, labels_dict, df):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for city in df['city'].unique():
        mask = df['city'] == city
        axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=CITY_COLORS[city], label=city, alpha=0.3, s=5)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 0].set_title('PCA by City')
    axes[0, 0].legend(markerscale=3)
    
    scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=labels_dict['kmeans'], cmap='viridis', alpha=0.3, s=5)
    axes[0, 1].set_xlabel(f'PC1')
    axes[0, 1].set_ylabel(f'PC2')
    axes[0, 1].set_title('K-Means Clusters')
    plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
    
    dbscan_labels = labels_dict['dbscan']
    noise_mask = dbscan_labels == -1
    axes[1, 0].scatter(X_pca[~noise_mask, 0], X_pca[~noise_mask, 1], 
                       c=dbscan_labels[~noise_mask], cmap='viridis', alpha=0.3, s=5)
    axes[1, 0].scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                       c='red', marker='x', s=20, label='Noise', alpha=0.5)
    axes[1, 0].set_xlabel(f'PC1')
    axes[1, 0].set_ylabel(f'PC2')
    axes[1, 0].set_title('DBSCAN Clusters')
    axes[1, 0].legend()
    
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                  c=labels_dict['gmm'], cmap='viridis', alpha=0.3, s=5)
    axes[1, 1].set_xlabel(f'PC1')
    axes[1, 1].set_ylabel(f'PC2')
    axes[1, 1].set_title('GMM Clusters')
    plt.colorbar(scatter, ax=axes[1, 1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cluster_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_visualization.png")

def plot_cluster_profiles(df, labels, method_name='K-Means'):
    df_plot = df.copy()
    df_plot['cluster'] = labels
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cluster_pm25 = df_plot.groupby('cluster')['pm25'].agg(['mean', 'std', 'count'])
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(cluster_pm25)))
    axes[0, 0].bar(cluster_pm25.index, cluster_pm25['mean'], 
                   yerr=cluster_pm25['std'], color=colors, capsize=5)
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('PM2.5 (µg/m³)')
    axes[0, 0].set_title(f'{method_name}: PM2.5 by Cluster')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    hourly = df_plot.groupby(['cluster', 'hour'])['pm25'].mean().unstack(level=0)
    hourly.plot(ax=axes[0, 1], marker='o', markersize=3)
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Mean PM2.5')
    axes[0, 1].set_title(f'{method_name}: Hourly Pattern by Cluster')
    axes[0, 1].legend(title='Cluster')
    axes[0, 1].grid(True, alpha=0.3)
    
    city_cluster = pd.crosstab(df_plot['city'], df_plot['cluster'], normalize='columns')
    city_cluster.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_xlabel('City')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].set_title(f'{method_name}: City Distribution by Cluster')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Cluster')
    
    season_cluster = pd.crosstab(df_plot['season'], df_plot['cluster'], normalize='columns')
    season_order = ['dry', 'transition', 'wet']
    season_cluster = season_cluster.reindex(season_order)
    season_cluster.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title(f'{method_name}: Season Distribution by Cluster')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'cluster_profiles_{method_name.lower().replace(" ", "_")}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: cluster_profiles_{method_name.lower().replace(' ', '_')}.png")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Total records: {len(df)}")
    
    print("\nPreparing features...")
    X_scaled, feature_cols, scaler = prepare_clustering_features(df)
    print(f"Features: {feature_cols}")
    
    print("\n" + "="*60)
    print("FINDING OPTIMAL K (using sampled silhouette for efficiency)")
    print("="*60)
    k_range, sse, silhouette_scores = find_optimal_k(X_scaled)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal k by silhouette: {optimal_k}")
    
    print("\n" + "="*60)
    print("TRAINING CLUSTERING MODELS")
    print("="*60)
    
    print(f"\n1. K-Means (k={optimal_k})...")
    kmeans, labels_kmeans = train_kmeans(X_scaled, n_clusters=optimal_k)
    eval_kmeans = evaluate_clustering(X_scaled, labels_kmeans, 'K-Means')
    print(f"   Silhouette: {eval_kmeans['Silhouette']:.4f}")
    
    print("\n2. DBSCAN (eps=0.8, min_samples=50)...")
    dbscan, labels_dbscan = train_dbscan(X_scaled)
    eval_dbscan = evaluate_clustering(X_scaled, labels_dbscan, 'DBSCAN')
    print(f"   Clusters: {eval_dbscan['N_Clusters']}, Noise: {eval_dbscan['Noise_Points']}")
    
    print(f"\n3. GMM (n_components={optimal_k})...")
    gmm, labels_gmm, probs_gmm = train_gmm(X_scaled, n_components=optimal_k)
    eval_gmm = evaluate_clustering(X_scaled, labels_gmm, 'GMM')
    print(f"   Silhouette: {eval_gmm['Silhouette']:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    results = [eval_kmeans, eval_dbscan, eval_gmm]
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS (K-Means)")
    print("="*60)
    df['cluster'] = labels_kmeans
    for c in range(optimal_k):
        cluster_data = df[df['cluster'] == c]
        print(f"\nCluster {c} (n={len(cluster_data)}):")
        print(f"  Mean PM2.5: {cluster_data['pm25'].mean():.1f} µg/m³")
        print(f"  Std PM2.5: {cluster_data['pm25'].std():.1f} µg/m³")
        print(f"  Unhealthy %: {cluster_data['is_unhealthy'].mean()*100:.1f}%")
        print(f"  Top city: {cluster_data['city'].mode().iloc[0]}")
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_elbow_and_silhouette(k_range, sse, silhouette_scores)
    
    labels_dict = {
        'kmeans': labels_kmeans,
        'dbscan': labels_dbscan,
        'gmm': labels_gmm
    }
    plot_cluster_visualization(X_scaled, labels_dict, df)
    plot_cluster_profiles(df, labels_kmeans, 'K-Means')
    
    results_file = os.path.join(FIGURES_DIR, 'clustering_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")
    
    df.to_csv(os.path.join(PROCESSED_DIR, 'data_with_clusters.csv'), index=False)
    print("Saved data with cluster labels")
    
    return results_df, labels_dict

if __name__ == '__main__':
    results, labels = main()
