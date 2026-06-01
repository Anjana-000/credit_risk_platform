import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import DATA_DIR

def run_eda_report():
    df = pd.read_csv(DATA_DIR / 'application_train.csv', nrows=10000)
    
    # Plot 1: Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='TARGET')
    plt.title('Class Imbalance: Defaults (1) vs Paid (0)')
    plt.savefig(DATA_DIR / 'target_dist.png')
    
    # Plot 2: Income vs Credit Amount
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='AMT_INCOME_TOTAL', y='AMT_CREDIT', hue='TARGET', alpha=0.5)
    plt.title('Income vs Credit Amount by Risk')
    plt.savefig(DATA_DIR / 'income_credit.png')
    
    return "EDA completed and plots saved to data directory."

if __name__ == "__main__":
    run_eda_report()