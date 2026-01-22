import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

def load_data():
    filepath = os.path.join(PROCESSED_DIR, 'combined_data.csv')
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    return df

def prepare_transaction_data(df):
    df_trans = df.copy()
    
    df_trans['pm25_level'] = pd.cut(df_trans['pm25'], 
                                     bins=[0, 12, 35.4, 55.4, 150.4, 500],
                                     labels=['good', 'moderate', 'usg', 'unhealthy', 'hazardous'])
    
    df_trans['time_period'] = pd.cut(df_trans['hour'],
                                      bins=[-1, 6, 12, 18, 24],
                                      labels=['night', 'morning', 'afternoon', 'evening'])
    
    df_trans['day_type'] = df_trans['is_weekend'].map({0: 'weekday', 1: 'weekend'})
    
    categorical_cols = ['city', 'season', 'time_period', 'day_type', 'pm25_level']
    
    for col in categorical_cols:
        df_trans[col] = df_trans[col].astype(str)
    
    return df_trans, categorical_cols

def create_one_hot_encoding(df, categorical_cols):
    transactions = []
    
    for _, row in df[categorical_cols].iterrows():
        transaction = [f"{col}={row[col]}" for col in categorical_cols]
        transactions.append(transaction)
    
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    return df_encoded

def mine_frequent_itemsets(df_encoded, min_support=0.05, method='apriori'):
    if method == 'apriori':
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.2):
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    
    rules = rules[rules['lift'] >= min_lift]
    
    rules = rules.sort_values('lift', ascending=False)
    
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    return rules

def filter_interesting_rules(rules):
    pollution_levels = ['pm25_level=unhealthy', 'pm25_level=hazardous', 
                        'pm25_level=usg', 'pm25_level=good']
    
    interesting_rules = rules[
        rules['consequents_str'].str.contains('pm25_level=', na=False) |
        rules['antecedents_str'].str.contains('pm25_level=', na=False)
    ].copy()
    
    return interesting_rules

def plot_support_confidence(rules, top_n=20):
    plot_rules = rules.head(top_n).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter = axes[0].scatter(plot_rules['support'], plot_rules['confidence'], 
                               c=plot_rules['lift'], cmap='viridis', 
                               s=100, alpha=0.7, edgecolors='black')
    axes[0].set_xlabel('Support')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Support vs Confidence (colored by Lift)')
    plt.colorbar(scatter, ax=axes[0], label='Lift')
    axes[0].grid(True, alpha=0.3)
    
    scatter = axes[1].scatter(plot_rules['support'], plot_rules['lift'], 
                               c=plot_rules['confidence'], cmap='plasma', 
                               s=100, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel('Support')
    axes[1].set_ylabel('Lift')
    axes[1].set_title('Support vs Lift (colored by Confidence)')
    plt.colorbar(scatter, ax=axes[1], label='Confidence')
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Lift=1 (baseline)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'association_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: association_scatter.png")

def plot_top_rules(rules, top_n=15):
    plot_rules = rules.head(top_n).copy()
    plot_rules['rule'] = plot_rules['antecedents_str'] + ' → ' + plot_rules['consequents_str']
    plot_rules['rule'] = plot_rules['rule'].str[:60] + '...'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(plot_rules['lift'] / plot_rules['lift'].max())
    
    bars = ax.barh(range(len(plot_rules)), plot_rules['lift'], color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(plot_rules)))
    ax.set_yticklabels(plot_rules['rule'], fontsize=9)
    ax.set_xlabel('Lift')
    ax.set_title(f'Top {top_n} Association Rules by Lift')
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Baseline (Lift=1)')
    
    for i, (lift, conf) in enumerate(zip(plot_rules['lift'], plot_rules['confidence'])):
        ax.text(lift + 0.05, i, f'Conf: {conf:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'top_association_rules.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: top_association_rules.png")

def plot_itemset_frequency(frequent_itemsets, top_n=20):
    single_items = frequent_itemsets[frequent_itemsets['length'] == 1].copy()
    single_items['item'] = single_items['itemsets'].apply(lambda x: list(x)[0])
    single_items = single_items.sort_values('support', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(single_items['support'] / single_items['support'].max())
    ax.barh(single_items['item'], single_items['support'], color=colors, edgecolor='black')
    ax.set_xlabel('Support')
    ax.set_title(f'Top {top_n} Frequent Items')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'frequent_items.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: frequent_items.png")

def analyze_rules_by_consequent(rules):
    print("\n" + "="*60)
    print("RULES BY CONSEQUENT (Air Quality Level)")
    print("="*60)
    
    for level in ['good', 'moderate', 'usg', 'unhealthy', 'hazardous']:
        level_rules = rules[rules['consequents_str'].str.contains(f'pm25_level={level}', na=False)]
        
        if len(level_rules) > 0:
            print(f"\n{level.upper()} Air Quality ({len(level_rules)} rules):")
            print("-" * 50)
            
            for _, rule in level_rules.head(3).iterrows():
                print(f"  IF {rule['antecedents_str']}")
                print(f"  THEN {rule['consequents_str']}")
                print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
                print()

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nPreparing transaction data...")
    df_trans, categorical_cols = prepare_transaction_data(df)
    print(f"Categorical columns: {categorical_cols}")
    
    print("\nCreating one-hot encoding...")
    df_encoded = create_one_hot_encoding(df_trans, categorical_cols)
    print(f"Encoded shape: {df_encoded.shape}")
    print(f"Items: {df_encoded.columns.tolist()}")
    
    print("\n" + "="*60)
    print("MINING FREQUENT ITEMSETS (Apriori)")
    print("="*60)
    
    frequent_itemsets = mine_frequent_itemsets(df_encoded, min_support=0.03, method='apriori')
    print(f"\nFound {len(frequent_itemsets)} frequent itemsets")
    print(f"Itemsets by length:")
    print(frequent_itemsets['length'].value_counts().sort_index())
    
    print("\n" + "="*60)
    print("GENERATING ASSOCIATION RULES")
    print("="*60)
    
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.4, min_lift=1.1)
    print(f"\nGenerated {len(rules)} rules")
    
    interesting_rules = filter_interesting_rules(rules)
    print(f"Interesting rules (involving pm25_level): {len(interesting_rules)}")
    
    print("\n" + "="*60)
    print("TOP 20 RULES BY LIFT")
    print("="*60)
    
    top_rules = rules.head(20)[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
    top_rules.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
    print("\n" + top_rules.to_string(index=False))
    
    analyze_rules_by_consequent(rules)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_support_confidence(rules, top_n=50)
    plot_top_rules(rules, top_n=15)
    plot_itemset_frequency(frequent_itemsets, top_n=20)
    
    rules_file = os.path.join(FIGURES_DIR, 'association_rules.csv')
    rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_csv(rules_file, index=False)
    print(f"\nSaved all rules to: {rules_file}")
    
    itemsets_file = os.path.join(FIGURES_DIR, 'frequent_itemsets.csv')
    frequent_itemsets.to_csv(itemsets_file, index=False)
    print(f"Saved frequent itemsets to: {itemsets_file}")
    
    return frequent_itemsets, rules

if __name__ == '__main__':
    frequent_itemsets, rules = main()
