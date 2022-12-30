import torch
import os
import logging
import argparse
import pandas as pd
import torch.nn.functional as F
import itertools
from torch_geometric.loader import DataLoader
from datasets.loaders import MutagenicityDataset
from models.graphs import ClassifierMutagenicity
from pathlib import Path
from utils.misc import set_random_seed
from utils.plots import robustness_plots
from captum.attr import IntegratedGradients, GradientShap
from utils.symmetries import GraphPermutation
from interpretability.robustness import graph_model_invariance, graph_explanation_equivariance,\
    graph_explanation_invariance, accuracy
from interpretability.feature import FeatureImportance, GraphFeatureAblation
from interpretability.example import GraphRepresentationSimilarity, GraphSimplEx, GraphTracIn, GraphInfluenceFunctions
from interpretability.concept import GraphCAR, GraphCAV
from torch.utils.data import Subset, RandomSampler


def train_mut_model(random_seed: int, latent_dim: int, batch_size: int, model_name: str = "model",
                    model_dir: Path = Path.cwd() / f"results/mut/", data_dir: Path = Path.cwd() / "datasets/mut") -> None:
    logging.info("Fitting the Mutagenicity classifiers")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_set = MutagenicityDataset(data_dir, train=True, random_seed=random_seed)
    test_set = MutagenicityDataset(data_dir, train=False, random_seed=random_seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    model = ClassifierMutagenicity(latent_dim)
    model.fit(device, train_loader, test_loader, model_dir, checkpoint_interval=20, patience=50, n_epoch=500)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mut/",
    data_dir: Path = Path.cwd() / "datasets/mut",
    N_samp: int = 1
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = MutagenicityDataset(data_dir, train=False, random_seed=random_seed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model_dir = model_dir / model_name
    model = ClassifierMutagenicity(latent_dim)
    model.load_metadata(model_dir)
    model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
    model.to(device).eval()
    attr_methods = {'Feature Ablation': GraphFeatureAblation, 'Gradient Shap': GradientShap,
                    'Integrated Gradients': IntegratedGradients}
    save_dir = model_dir/'feature_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    graph_perm = GraphPermutation()
    metrics = []
    logging.info(f'Now working with Mutagenicity classifier')
    model_inv = graph_model_invariance(model, graph_perm, test_loader, device, N_samp=N_samp)
    logging.info(f'Model invariance: {torch.mean(model_inv).item():.3g}')
    for attr_name in attr_methods:
        logging.info(f'Now working with {attr_name}')
        feat_importance = FeatureImportance(attr_methods[attr_name](model))
        explanation_equiv = graph_explanation_equivariance(feat_importance, graph_perm, test_loader, device, N_samp=N_samp)
        logging.info(f'Explanation equivariance: {torch.mean(explanation_equiv):.3g}')
        for inv, equiv in zip(model_inv, explanation_equiv):
            metrics.append(['GNN', attr_name, inv.item(), equiv.item()])
    metrics_df = pd.DataFrame(data=metrics,
                              columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Equivariance'])
    metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
    if plot:
        robustness_plots(save_dir, 'mut', 'feature_importance')


def example_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mut/",
    data_dir: Path = Path.cwd() / "datasets/mut",
    n_train: int = 100,
    recursion_depth: int = 100,
    N_samp: int = 1
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = MutagenicityDataset(data_dir, train=True, random_seed=random_seed)
    train_subset = Subset(train_set, torch.randperm(len(train_set))[:n_train])
    train_loader = DataLoader(train_subset, n_train, shuffle=False)  # Used to sample training graphs to attribute
    data_train = next(iter(train_loader))
    data_train = data_train.to(device)
    train_loader = DataLoader(train_subset, 1, shuffle=False)  # Loss-based attributions require batch size of 1
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=recursion_depth*batch_size)
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_set = MutagenicityDataset(data_dir, train=False, random_seed=random_seed)
    test_loader = DataLoader(test_set, 1, shuffle=False)
    models = {'GNN': ClassifierMutagenicity(latent_dim)}
    attr_methods = {'Influence Functions': GraphInfluenceFunctions, 'TracIn': GraphTracIn,
                    'Representation Similarity': GraphRepresentationSimilarity, 'SimplEx': GraphSimplEx,}
    model_dir = model_dir/model_name
    save_dir = model_dir/'example_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    graph_permutation = GraphPermutation()
    metrics = []
    for model_type in models:
        logging.info(f'Now working with {model_type} classifier')
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = graph_model_invariance(model, graph_permutation, test_loader, device, N_samp=N_samp)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        model_layers = {'Lin1': model.lin1}
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
            if attr_name in {'TracIn', 'Influence Functions'}:
                ex_importance = attr_methods[attr_name](model, train_loader, train_sampler=train_loader_replacement,
                                                        loss_function=F.nll_loss, save_dir=save_dir/model.name,
                                                        recursion_depth=recursion_depth, device=device)
                explanation_inv = graph_explanation_invariance(ex_importance, graph_permutation, test_loader, device, N_samp=N_samp)
                for inv_model, inv_expl in zip(model_inv, explanation_inv):
                    metrics.append([model_type, attr_name, inv_model.item(), inv_expl.item()])
                logging.info(f'Explanation invariance: {torch.mean(explanation_inv):.3g}')
            else:
                for layer_name in model_layers:
                    ex_importance = attr_methods[attr_name](model, data_train, layer=model_layers[layer_name])
                    explanation_inv = graph_explanation_invariance(ex_importance, graph_permutation, test_loader, device, N_samp=N_samp)
                    ex_importance.remove_hook()
                    for inv_model, inv_expl in zip(model_inv, explanation_inv):
                        metrics.append([model_type, f'{attr_name}-{layer_name}', inv_model.item(), inv_expl.item()])
                    logging.info(f'Explanation invariance for {layer_name}: {torch.mean(explanation_inv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        robustness_plots(save_dir, 'mut', 'example_importance')


def concept_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mut/",
    data_dir: Path = Path.cwd() / "datasets/mut",
    concept_set_size: int = 500,
    N_samp: int = 1
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = MutagenicityDataset(data_dir, train=True, random_seed=random_seed)
    test_set = MutagenicityDataset(data_dir, train=False, random_seed=random_seed)
    train_set.generate_concept_dataset(0, concept_set_size)
    test_loader = DataLoader(test_set, 1, shuffle=False)
    models = {'GNN': ClassifierMutagenicity(latent_dim)}
    attr_methods = {'CAR': GraphCAR, 'CAV': GraphCAV,}
    model_dir = model_dir/model_name
    save_dir = model_dir/'concept_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    graph_permutation = GraphPermutation()
    metrics = []
    for model_type in models:
        logging.info(f'Now working with {model_type} classifier')
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = graph_model_invariance(model, graph_permutation, test_loader, device, N_samp=N_samp)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        model_layers = {'Lin1': model.lin1}
        for layer_name, attr_name in itertools.product(model_layers, attr_methods):
            logging.info(f'Now working with {attr_name} explainer on layer {layer_name}')
            conc_importance = attr_methods[attr_name](model, train_set, n_classes=2, layer=model_layers[layer_name])
            conc_importance.fit(device, concept_set_size, batch_size=batch_size)
            concept_acc = conc_importance.concept_accuracy(test_set, device, batch_size=batch_size)
            for concept_name in concept_acc:
                logging.info(f'Concept {concept_name} accuracy: {concept_acc[concept_name]:.2g}')
            explanation_inv = graph_explanation_invariance(conc_importance, graph_permutation, test_loader, device,
                                                           similarity=accuracy, N_samp=N_samp)
            conc_importance.remove_hook()
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append([model_type, f'{attr_name}-{layer_name}', inv_model.item(), inv_expl.item()])
            logging.info(f'Explanation invariance: {torch.mean(explanation_inv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        robustness_plots(save_dir, 'mut', 'concept_importance')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--N_samp", type=int, default=50)
    args = parser.parse_args()
    model_name = f"gnn{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_mut_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.seed, args.latent_dim, args.plot, model_name, N_samp=args.N_samp)
        case 'example_importance':
            example_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, N_samp=args.N_samp)
        case 'concept_importance':
            concept_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, N_samp=args.N_samp)
        case other:
            raise ValueError('Invalid experiment name.')
