import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import logging
import argparse
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette('colorblind')


def robustness_plots(plot_dir: Path, dataset: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    for model_type in metrics_df['Model Type'].unique():
        sub_df = metrics_df[metrics_df['Model Type'] == model_type]
        y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
        ax = sns.boxplot(sub_df, x='Explanation', y=y, showfliers=False)
        wrap_labels(ax, 10)
        plt.tight_layout()
        plt.savefig(plot_dir/f'{dataset}_{model_type.lower().replace(" ", "_")}.pdf')
        plt.close()


def relaxing_invariance_plots(plot_dir: Path, dataset: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
    plot_df = metrics_df.groupby(['Model Type', 'Explanation']).mean()
    plot_df[['Model Invariance CI', f'{y} CI']] = 2 * metrics_df.groupby(['Model Type', 'Explanation']).sem()
    ax = sns.scatterplot(plot_df, x='Model Invariance', y=y, hue='Model Type',
                         style='Explanation',  markers=['o', 'v', 's', '*', 'd'])
    plt.errorbar(x=plot_df['Model Invariance'], y=plot_df[y],
                 xerr=plot_df['Model Invariance CI'], yerr=plot_df[f'{y} CI'],
                 ecolor='k', linestyle='')
    plt.xscale('linear')
    plt.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
    plt.tight_layout()
    plt.savefig(plot_dir/f'{dataset}_relaxing_invariance.pdf')
    plt.close()


def wrap_labels(ax, width, break_long_words=False, do_y: bool = False) -> None:
    """
    Break labels in several lines in a figure
    Args:
        ax: figure axes
        width: maximal number of characters per line
        break_long_words: if True, allow breaks in the middle of a word
        do_y: if True, apply the function to the y axis as well
    Returns:
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    if do_y:
        labels = []
        for label in ax.get_yticklabels():
            text = label.get_text()
            labels.append(textwrap.fill(text, width=width,
                                        break_long_words=break_long_words))
        ax.set_yticklabels(labels, rotation=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="relax_invariance")
    parser.add_argument("--dataset", type=str, default="ecg")
    parser.add_argument("--model", type=str, default="cnn32_seed42")
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()
    plot_path = Path.cwd()/f"results/{args.dataset}/{args.model}"
    logging.info(f"Saving {args.name} plot for {args.dataset} in {str(plot_path)}")
    match args.name:
        case 'robustness':
            robustness_plots(plot_path, args.dataset)
        case 'relax_invariance':
            relaxing_invariance_plots(plot_path, args.dataset)
        case other:
            raise ValueError("Unknown plot name")
