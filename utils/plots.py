import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette('colorblind')


def robustness_plots(plot_dir: Path) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    for model_type in metrics_df['Model Type'].unique():
        sub_df = metrics_df[metrics_df['Model Type'] == model_type]
        y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
        sns.boxplot(sub_df, x='Explanation', y=y)
        plt.savefig(plot_dir/f'{model_type.lower().replace(" ", "_")}.pdf')
        plt.close()