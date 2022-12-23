import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import logging
import argparse
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette('colorblind')
markers = ["o", "s", "X", "D", "v"]


def robustness_plots(plot_dir: Path, dataset: str, experiment_name: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    for model_type in metrics_df['Model Type'].unique():
        sub_df = metrics_df[metrics_df['Model Type'] == model_type]
        y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
        ax = sns.boxplot(sub_df, x='Explanation', y=y, showfliers=False)
        wrap_labels(ax, 10)
        plt.tight_layout()
        plt.savefig(plot_dir/f'{experiment_name}_{dataset}_{model_type.lower().replace(" ", "_")}.pdf')
        plt.close()


def relaxing_invariance_plots(plot_dir: Path, dataset: str, experiment_name: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
    plot_df = metrics_df.groupby(['Model Type', 'Explanation']).mean()
    plot_df[['Model Invariance CI', f'{y} CI']] = 2 * metrics_df.groupby(['Model Type', 'Explanation']).sem()
    sns.scatterplot(plot_df, x='Model Invariance', y=y, hue='Model Type', edgecolor="black", alpha=.5,
                         style='Explanation',  markers=markers[:metrics_df['Explanation'].nunique()])
    plt.errorbar(x=plot_df['Model Invariance'], y=plot_df[y],
                 xerr=plot_df['Model Invariance CI'], yerr=plot_df[f'{y} CI'],
                 ecolor='black', elinewidth=.7, linestyle='', capsize=.7, capthick=.7)
    plt.xscale('linear')
    plt.axline((0, 0), slope=1, color="gray", linestyle='dotted')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(plot_dir/f'{experiment_name}_{dataset}_relaxing_invariance.pdf')
    plt.close()


def mc_convergence_plot(plot_dir: Path, dataset: str, experiment_name: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    for estimator_name in metrics_df['Estimator Name'].unique():
        metrics_subdf = metrics_df[metrics_df['Estimator Name'] == estimator_name]
        x = metrics_subdf['Number of MC Samples']
        y = metrics_subdf['Estimator Value']
        ci = 2*metrics_subdf['Estimator SEM']
        plt.plot(x, y, label=estimator_name)
        plt.fill_between(x, y-ci, y+ci, alpha=0.2)
    plt.legend()
    plt.xlabel(r'$N_{\mathrm{samp}}$')
    plt.ylabel('Monte Carlo Estimator')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{experiment_name}_{dataset}.pdf')
    plt.close()


def understanding_randomness_plots(plot_dir: Path, dataset: str) -> None:
    data_df = pd.read_csv(plot_dir / 'data.csv')
    sub_df = data_df[data_df['Baseline'] == False]
    print(sub_df)
    sns.kdeplot(data=data_df, x='y1', y='y2', hue='Model Type', fill=True)
    for model_type in data_df['Model Type'].unique():
        baseline = data_df[(data_df['Model Type'] == model_type) & (data_df['Baseline'] == True)]
        plt.plot(baseline['y1'], baseline['y2'], marker="x", linewidth=0, label=f'Baseline {model_type}')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.legend()
    plt.show()


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
    parser.add_argument("--experiment_name", type=str, default="feature_importance")
    parser.add_argument("--plot_name", type=str, default="relax_invariance")
    parser.add_argument("--dataset", type=str, default="ecg")
    parser.add_argument("--model", type=str, default="cnn32_seed42")
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()
    plot_path = Path.cwd()/f"results/{args.dataset}/{args.model}/{args.experiment_name}"
    logging.info(f"Saving {args.plot_name} plot for {args.dataset} in {str(plot_path)}")
    match args.plot_name:
        case 'robustness':
            robustness_plots(plot_path, args.dataset, args.experiment_name)
        case 'relax_invariance':
            relaxing_invariance_plots(plot_path, args.dataset, args.experiment_name)
        case 'mc_convergence':
            mc_convergence_plot(plot_path, args.dataset, args.experiment_name)
        case other:
            raise ValueError("Unknown plot name")
