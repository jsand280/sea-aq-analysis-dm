import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, 
                             precision_score, recall_score, accuracy_score)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def load_data():
    filepath = os.path.join(PROCESSED_DIR, 'combined_data.csv')
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    return df

def prepare_features(df):
    """
    FORECASTING TASK: Predict whether NEXT HOUR will be unhealthy
    Features: Only PAST information (no leakage)
    """
    df_clean = df.dropna(subset=['target_next_hour']).copy()
    
    feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                    'pm25_lag_1h', 'pm25_lag_24h', 
                    'pm25_rolling_mean_24h', 'pm25_rolling_std_24h']
    
    city_dummies = pd.get_dummies(df_clean['city'], prefix='city')
    
    X = pd.concat([df_clean[feature_cols].reset_index(drop=True), 
                   city_dummies.reset_index(drop=True)], axis=1)
    y = df_clean['target_next_hour'].astype(int).reset_index(drop=True)
    
    X = X.fillna(X.median())
    
    return X, y, X.columns.tolist()

def train_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        min_samples_split=100,
        class_weight='balanced',
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]
    return dt, y_pred, y_prob

def train_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]
    return knn, y_pred, y_prob

def train_svm(X_train_scaled, X_test_scaled, y_train, y_test):
    """Linear SVM with calibration for probability estimates (scalable)"""
    base_svm = LinearSVC(
        C=1.0,
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    )
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]
    return svm, y_pred, y_prob

def train_naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]
    return nb, y_pred, y_prob

def evaluate_model(y_test, y_pred, y_prob, model_name):
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    return metrics

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Healthy', 'Unhealthy'],
                    yticklabels=['Healthy', 'Unhealthy'])
        axes[idx].set_title(f'{name}\nAccuracy: {data["metrics"]["Accuracy"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices: Predicting Next-Hour Air Quality', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices.png")

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(10, 8))
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6']
    
    for idx, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
        auc = data['metrics']['ROC-AUC']
        plt.plot(fpr, tpr, color=colors[idx], linewidth=2, 
                 label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Next-Hour Air Quality Prediction')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: roc_curves.png")

def plot_metrics_comparison(all_metrics):
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.set_index('Model', inplace=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_df[metrics_to_plot].plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('Classification Metrics Comparison')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    auc_scores = metrics_df['ROC-AUC']
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6']
    bars = axes[1].bar(auc_scores.index, auc_scores.values, color=colors)
    axes[1].set_title('ROC-AUC Comparison')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_ylim(0.5, 1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, auc_scores.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_comparison.png")

def plot_feature_importance(dt, feature_names):
    importance = dt.feature_importances_
    indices = np.argsort(importance)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices][::-1], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Decision Tree: Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")

def cross_validate_models(X, y, scaler):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, 
                                                 class_weight='balanced', random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000, random_state=42), cv=3),
        'Naive Bayes': GaussianNB()
    }
    
    cv_results = {}
    for name, model in models.items():
        if name in ['k-NN', 'Linear SVM']:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"  {name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("="*60)
    print("CLASSIFICATION: Predicting Next-Hour Air Quality")
    print("Task: Forecast whether the NEXT hour will be unhealthy")
    print("="*60)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Total records: {len(df)}")
    
    print("\nPreparing features (NO LEAKAGE - using only past data)...")
    X, y, feature_names = prepare_features(df)
    print(f"Features: {feature_names}")
    print(f"Samples: {len(X)}, Target distribution: {y.value_counts().to_dict()}")
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    results = {}
    all_metrics = []
    
    print("\n1. Decision Tree...")
    dt, y_pred_dt, y_prob_dt = train_decision_tree(X_train, X_test, y_train, y_test)
    metrics_dt = evaluate_model(y_test, y_pred_dt, y_prob_dt, 'Decision Tree')
    results['Decision Tree'] = {'model': dt, 'y_pred': y_pred_dt, 'y_prob': y_prob_dt, 'metrics': metrics_dt}
    all_metrics.append(metrics_dt)
    print(f"   F1: {metrics_dt['F1']:.4f}, AUC: {metrics_dt['ROC-AUC']:.4f}")
    
    print("\n2. k-Nearest Neighbors...")
    knn, y_pred_knn, y_prob_knn = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    metrics_knn = evaluate_model(y_test, y_pred_knn, y_prob_knn, 'k-NN')
    results['k-NN'] = {'model': knn, 'y_pred': y_pred_knn, 'y_prob': y_prob_knn, 'metrics': metrics_knn}
    all_metrics.append(metrics_knn)
    print(f"   F1: {metrics_knn['F1']:.4f}, AUC: {metrics_knn['ROC-AUC']:.4f}")
    
    print("\n3. Linear SVM (scalable)...")
    svm, y_pred_svm, y_prob_svm = train_svm(X_train_scaled, X_test_scaled, y_train, y_test)
    metrics_svm = evaluate_model(y_test, y_pred_svm, y_prob_svm, 'Linear SVM')
    results['Linear SVM'] = {'model': svm, 'y_pred': y_pred_svm, 'y_prob': y_prob_svm, 'metrics': metrics_svm}
    all_metrics.append(metrics_svm)
    print(f"   F1: {metrics_svm['F1']:.4f}, AUC: {metrics_svm['ROC-AUC']:.4f}")
    
    print("\n4. Naive Bayes...")
    nb, y_pred_nb, y_prob_nb = train_naive_bayes(X_train, X_test, y_train, y_test)
    metrics_nb = evaluate_model(y_test, y_pred_nb, y_prob_nb, 'Naive Bayes')
    results['Naive Bayes'] = {'model': nb, 'y_pred': y_pred_nb, 'y_prob': y_prob_nb, 'metrics': metrics_nb}
    all_metrics.append(metrics_nb)
    print(f"   F1: {metrics_nb['F1']:.4f}, AUC: {metrics_nb['ROC-AUC']:.4f}")
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-Fold)")
    print("="*60)
    cv_results = cross_validate_models(X, y, scaler)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))
    
    best_model = max(all_metrics, key=lambda x: x['F1'])
    print(f"\nBest model by F1: {best_model['Model']} ({best_model['F1']:.4f})")
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_metrics_comparison(all_metrics)
    plot_feature_importance(dt, feature_names)
    
    results_file = os.path.join(FIGURES_DIR, 'classification_results.csv')
    metrics_df.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")
    
    return results, all_metrics

if __name__ == '__main__':
    results, metrics = main()
