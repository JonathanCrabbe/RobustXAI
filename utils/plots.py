import json
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import logging
import argparse
import networkx as nx
from pathlib import Path


sns.set_style("whitegrid")
sns.set_palette('colorblind')
markers = ["o", "s", "X", "D", "v", "p"]


def single_robustness_plots(plot_dir: Path, dataset: str, experiment_name: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    for model_type in metrics_df['Model Type'].unique():
        sub_df = metrics_df[metrics_df['Model Type'] == model_type]
        y = 'Explanation Equivariance' if 'Explanation Equivariance' in metrics_df.columns else 'Explanation Invariance'
        ax = sns.boxplot(sub_df, x='Explanation', y=y, showfliers=False)
        wrap_labels(ax, 10)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(plot_dir/f'{experiment_name}_{dataset}_{model_type.lower().replace(" ", "_")}.pdf')
        plt.close()


def global_robustness_plots(experiment_name: str) -> None:
    with open(Path.cwd()/"results_dir.json") as f:
        path_dic = json.load(f)
    global_df = []
    for dataset in path_dic:
        dataset_df = pd.read_csv(Path.cwd()/path_dic[dataset]/experiment_name/"metrics.csv")
        dataset_df['Dataset'] = [dataset]*len(dataset_df)
        global_df.append(dataset_df)
    global_df = pd.concat(global_df)
    rename_dic = {'SimplEx-Lin1': 'SimplEx-Inv', 'SimplEx-Conv3': 'SimplEx-Equiv',
                  'Representation Similarity-Lin1': 'Rep. Similar-Inv', 'Representation Similarity-Conv3': 'Rep. Similar-Equiv',
                  'CAR-Lin1': 'CAR-Inv', 'CAR-Conv3': 'CAR-Equiv', 'CAV-Lin1': 'CAV-Inv', 'CAV-Conv3': 'CAV-Equiv'}
    global_df = global_df.replace(rename_dic)
    global_df = global_df[(global_df['Model Type'] == 'All-CNN') | (global_df['Model Type'] == 'GNN')]
    y = 'Explanation Equivariance' if 'Explanation Equivariance' in global_df.columns else 'Explanation Invariance'
    ax = sns.boxplot(global_df, x='Dataset', hue='Explanation', y=y, showfliers=False)
    wrap_labels(ax, 10)
    plt.ylim(-1.1, 1.1)
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        patch.set_facecolor('None')

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers

    # Also fix the legend
    for legpatch in ax.legend_.get_patches():
        col = legpatch.get_facecolor()
        legpatch.set_edgecolor(col)
        legpatch.set_facecolor('None')
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(Path.cwd()/f'results/{experiment_name}_global_robustness.pdf')
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


def enforce_invariance_plot(plot_dir: Path, dataset: str) -> None:
    metrics_df = pd.read_csv(plot_dir/'metrics.csv')
    sns.lineplot(metrics_df, x='N_inv', y='Explanation Invariance', hue='Explanation')
    plt.legend()
    plt.xlabel(r'$N_{\mathrm{inv}}$')
    plt.tight_layout()
    plt.savefig(plot_dir / f'enforce_invariance_{dataset}.pdf')
    plt.close()


def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
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
            single_robustness_plots(plot_path, args.dataset, args.experiment_name)
        case 'global_robustness':
            global_robustness_plots(args.experiment_name)
        case 'relax_invariance':
            relaxing_invariance_plots(plot_path, args.dataset, args.experiment_name)
        case 'mc_convergence':
            mc_convergence_plot(plot_path, args.dataset, args.experiment_name)
        case other:
            raise ValueError("Unknown plot name")
