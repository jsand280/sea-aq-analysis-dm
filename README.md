# Southeast Asia Air Quality Analysis

Analyzed PM2.5 data from Bangkok, Ho Chi Minh City, Kuala Lumpur, and Singapore (9,170 hourly observations) to predict unhealthy air quality levels and uncover pollution patterns using classification, clustering, and anomaly detection.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib
**Data:** OpenAQ API (Jan-July 2023)

## Key Results

- **97.7% accuracy** predicting next-hour air quality using KNN (all models achieved >0.97 ROC-AUC)
- **Ho Chi Minh City has 220x more pollution anomalies than Singapore** (11.06% vs 0.05% of readings)
- **8 distinct pollution patterns** identified through K-Means clustering
- Association rules found Singapore + wet season strongly predicts good air quality (lift: 2.54)

## Sample Output

![ROC Curves](figures/roc_curves.png)

![Cluster Visualization](figures/cluster_visualization.png)

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
