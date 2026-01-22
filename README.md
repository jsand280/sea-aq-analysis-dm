# Southeast Asia Air Quality Analysis

Analyzed PM2.5 data from Bangkok, Ho Chi Minh City, Kuala Lumpur, and Singapore (9,170 hourly observations) to predict unhealthy air quality levels and uncover pollution patterns using classification, clustering, and anomaly detection.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib
**Data:** OpenAQ API (Jan-July 2023)

## Key Results

- **97.7% accuracy** predicting next-hour air quality using KNN (all models achieved >0.97 ROC-AUC)
- **Ho Chi Minh City has 220x more pollution anomalies than Singapore** (11.06% vs 0.05% of readings)
- **8 distinct pollution patterns** identified through K-Means clustering
- Association rules found Singapore + wet season strongly predicts good air quality (lift: 2.54)

## Visualizations

### Exploratory Data Analysis

PM2.5 levels vary dramatically across cities. Ho Chi Minh City shows the highest variability (SD: 101.5 µg/m³) with extreme readings up to 985 µg/m³, while Singapore maintains consistently clean air (max: 27 µg/m³).

![PM2.5 Distribution](figures/pm25_distribution.png)

Temporal patterns reveal seasonal and daily trends. The dry season (Nov-Feb) shows elevated pollution, particularly in Bangkok which exceeds the unhealthy threshold (35.4 µg/m³) during this period.

![Time Series](figures/time_series.png)

### Classification: Next-Hour Air Quality Prediction

All four classifiers achieved ROC-AUC > 0.97, demonstrating that past PM2.5 data reliably predicts next-hour air quality. KNN achieved the best F1 score (0.873) due to air pollution's gradual change—similar past conditions lead to similar future outcomes.

![ROC Curves](figures/roc_curves.png)

The confusion matrices show the precision-recall tradeoff: Decision Tree catches 97.2% of unhealthy events but has more false alarms, while KNN balances both with 95.4% precision and 80.4% recall.

![Confusion Matrices](figures/confusion_matrices.png)

### Clustering: Identifying Pollution Patterns

Silhouette analysis identified k=8 as optimal, revealing 8 distinct pollution patterns across the dataset. The elbow curve shows diminishing returns beyond this point.

![Clustering Elbow](figures/clustering_elbow.png)

PCA visualization shows clear geographic separation; Singapore and Kuala Lumpur cluster together (clean air), while Bangkok and Ho Chi Minh City form distinct high-pollution clusters.

![Cluster Visualization](figures/cluster_visualization.png)

### Anomaly Detection & Dimensionality Reduction

Three methods (Z-score, LOF, One-Class SVM) detect anomalies with consensus requiring agreement from at least 2. Ho Chi Minh City has 220x more anomalies than Singapore (11.06% vs 0.05%), reflecting its extreme pollution variability.

![Anomaly Comparison](figures/anomaly_comparison.png)

PCA reveals that 3 components capture 90% of variance. PC1 strongly correlates with PM2.5 levels, while PC2 captures temporal patterns (hour, day of week).

![PCA Analysis](figures/pca_analysis.png)

### Association Rules: Condition-Pollution Patterns

Apriori mining found strong associations between location, season, and air quality. Singapore consistently associates with good air quality regardless of season (lift: 2.54-2.71), providing actionable insights for public health warnings.

![Top Association Rules](figures/top_association_rules.png)

## Methods

| Task | Techniques |
|------|------------|
| Classification | KNN, Decision Tree, Linear SVM, Naive Bayes |
| Clustering | K-Means, DBSCAN, GMM |
| Other | Apriori rules, Z-score/LOF/One-Class SVM anomaly detection, PCA |

## Quick Start

```bash
pip install -r requirements.txt
cd scripts && python download_data.py && python run_all.py
```
