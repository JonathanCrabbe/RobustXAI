import argparse
import itertools
import logging
import os
from math import sqrt
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from captum.attr import (FeatureAblation, FeaturePermutation, GradientShap,
                         IntegratedGradients)
from torch.utils.data import DataLoader, RandomSampler, Subset

from datasets.loaders import ModelNet40Dataset
from interpretability.concept import CAR, CAV
from interpretability.example import (InfluenceFunctions,
                                      RepresentationSimilarity, SimplEx,
                                      TracIn)
from interpretability.feature import FeatureImportance
from interpretability.robustness import (accuracy, cos_similarity,
                                         explanation_equivariance,
                                         explanation_invariance,
                                         model_invariance)
from models.sets import ClassifierModelNet40
from utils.misc import set_random_seed
from utils.plots import mc_convergence_plot, single_robustness_plots
from utils.symmetries import SetPermutation


def train_mnet_model(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
) -> None:
    logging.info("Fitting the ModelNet40 classifier")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = ClassifierModelNet40(latent_dim=latent_dim, name=model_name)
    train_set = ModelNet40Dataset(data_dir, train=True, random_seed=random_seed)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    model.fit(device, train_loader, test_loader, model_dir, checkpoint_interval=10)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
    n_test: int = 1000,
    N_samp: int = 50,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {"Deep-Set": ClassifierModelNet40(latent_dim=latent_dim, name=model_name)}
    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Feature Permutation": FeaturePermutation,
        "Feature Ablation": FeatureAblation,
    }
    model_dir = model_dir / model_name
    save_dir = model_dir / "feature_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    permutation = SetPermutation()
    metrics = []
    for model_type in models:
        logging.info(f"Now working with {model_type} classifier")
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(
            model, permutation, test_loader, device, N_samp=N_samp
        )
        logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
        for attr_name in attr_methods:
            logging.info(f"Now working with {attr_name} explainer")
            feat_importance = FeatureImportance(attr_methods[attr_name](model))
            explanation_equiv = explanation_equivariance(
                feat_importance, permutation, test_loader, device, N_samp=N_samp
            )
            for inv, equiv in zip(model_inv, explanation_equiv):
                metrics.append([model_type, attr_name, inv.item(), equiv.item()])
            logging.info(
                f"Explanation equivariance: {torch.mean(explanation_equiv):.3g}"
            )
    metrics_df = pd.DataFrame(
        data=metrics,
        columns=[
            "Model Type",
            "Explanation",
            "Model Invariance",
            "Explanation Equivariance",
        ],
    )
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "mnet", "feature_importance")


def example_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
    n_test: int = 1000,
    n_train: int = 100,
    recursion_depth: int = 100,
    N_samp: int = 50,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ModelNet40Dataset(data_dir, train=True, random_seed=random_seed)
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {"Deep-Set": ClassifierModelNet40(latent_dim, name=model_name)}
    attr_methods = {
        "Representation Similarity": RepresentationSimilarity,
        "SimplEx": SimplEx,
        "TracIn": TracIn,
        "Influence Functions": InfluenceFunctions,
    }
    model_dir = model_dir / model_name
    save_dir = model_dir / "example_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    permutation = SetPermutation()
    metrics = []
    for model_type in models:
        logging.info(f"Now working with {model_type} classifier")
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(
            model, permutation, test_loader, device, N_samp=N_samp
        )
        logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
        model_layers = {"Phi": model.phi[2], "Rho": model.ro[2]}
        for attr_name in attr_methods:
            logging.info(f"Now working with {attr_name} explainer")
            model.load_state_dict(
                torch.load(model_dir / f"{model.name}.pt"), strict=False
            )
            if attr_name in {"TracIn", "Influence Functions"}:
                ex_importance = attr_methods[attr_name](
                    model,
                    X_train,
                    Y_train=Y_train,
                    train_loader=train_loader_replacement,
                    loss_function=nn.CrossEntropyLoss(),
                    save_dir=save_dir / model.name,
                    recursion_depth=recursion_depth,
                )
                explanation_inv = explanation_invariance(
                    ex_importance, permutation, test_loader, device, N_samp=N_samp
                )
                for inv_model, inv_expl in zip(model_inv, explanation_inv):
                    metrics.append(
                        [model_type, attr_name, inv_model.item(), inv_expl.item()]
                    )
                logging.info(
                    f"Explanation invariance: {torch.mean(explanation_inv):.3g}"
                )
            else:
                for layer_name in model_layers:
                    ex_importance = attr_methods[attr_name](
                        model, X_train, Y_train=Y_train, layer=model_layers[layer_name]
                    )
                    explanation_inv = explanation_invariance(
                        ex_importance, permutation, test_loader, device, N_samp=N_samp
                    )
                    ex_importance.remove_hook()
                    for inv_model, inv_expl in zip(model_inv, explanation_inv):
                        metrics.append(
                            [
                                model_type,
                                f"{attr_name}-{layer_name}",
                                inv_model.item(),
                                inv_expl.item(),
                            ]
                        )
                    logging.info(
                        f"Explanation invariance for {layer_name}: {torch.mean(explanation_inv):.3g}"
                    )
    metrics_df = pd.DataFrame(
        data=metrics,
        columns=[
            "Model Type",
            "Explanation",
            "Model Invariance",
            "Explanation Invariance",
        ],
    )
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "mnet", "example_importance")


def concept_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
    n_test: int = 1000,
    concept_set_size: int = 500,
    N_samp: int = 50,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ModelNet40Dataset(data_dir, train=True, random_seed=random_seed)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {"Deep-Set": ClassifierModelNet40(latent_dim, name=model_name)}
    attr_methods = {"CAV": CAV, "CAR": CAR}
    model_dir = model_dir / model_name
    save_dir = model_dir / "concept_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    permutation = SetPermutation()
    metrics = []
    for model_type in models:
        logging.info(f"Now working with {model_type} classifier")
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(
            model, permutation, test_loader, device, N_samp=N_samp
        )
        logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
        model_layers = {"Phi": model.phi[2], "Rho": model.ro[2]}
        for layer_name, attr_name in itertools.product(model_layers, attr_methods):
            logging.info(
                f"Now working with {attr_name} explainer on layer {layer_name}"
            )
            conc_importance = attr_methods[attr_name](
                model, train_set, n_classes=2, layer=model_layers[layer_name]
            )
            conc_importance.fit(device, concept_set_size)
            concept_acc = conc_importance.concept_accuracy(
                test_set, device, concept_set_size=concept_set_size
            )
            for concept_name in concept_acc:
                logging.info(
                    f"Concept {concept_name} accuracy: {concept_acc[concept_name]:.2g}"
                )
            explanation_inv = explanation_invariance(
                conc_importance,
                permutation,
                test_loader,
                device,
                similarity=accuracy,
                N_samp=N_samp,
            )
            conc_importance.remove_hook()
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append(
                    [
                        model_type,
                        f"{attr_name}-{layer_name}",
                        inv_model.item(),
                        inv_expl.item(),
                    ]
                )
            logging.info(f"Explanation invariance: {torch.mean(explanation_inv):.3g}")
    metrics_df = pd.DataFrame(
        data=metrics,
        columns=[
            "Model Type",
            "Explanation",
            "Model Invariance",
            "Explanation Invariance",
        ],
    )
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "mnet", "concept_importance")


def mc_convergence(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
    n_train: int = 100,
    n_test: int = 1000,
    N_samp_max: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ModelNet40Dataset(data_dir, train=True, random_seed=random_seed)
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, _ = next(iter(train_loader))
    X_train = X_train.to(device)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    model_dir = model_dir / model_name
    model = ClassifierModelNet40(latent_dim=latent_dim, name=model_name)
    model.load_metadata(model_dir)
    model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
    model.to(device).eval()
    permutation = SetPermutation()
    save_dir = model_dir / "mc_convergence"
    mc_estimators = {
        "Feature Permutation Equivariance": (
            FeaturePermutation,
            explanation_equivariance,
            cos_similarity,
        ),
        "Gradient Shap Equivariance": (
            GradientShap,
            explanation_equivariance,
            cos_similarity,
        ),
        "SimplEx-Equiv Invariance": (SimplEx, explanation_invariance, cos_similarity),
        "CAV-Equiv Invariance": (CAV, explanation_invariance, accuracy),
        "CAR-Equiv Invariance": (CAR, explanation_invariance, accuracy),
    }
    if not save_dir.exists():
        os.makedirs(save_dir)
    data = []
    for estimator_name in mc_estimators:
        logging.info(f"Computing Monte Carlo estimators for {estimator_name}")
        attr_method, metric, similarity = mc_estimators[estimator_name]
        if attr_method in {FeaturePermutation, GradientShap}:
            explanation = FeatureImportance(attr_method(model))
        elif attr_method in {CAR, CAV}:
            explanation = attr_method(model, train_set, model.phi[2], n_classes=40)
            explanation.fit(device, 500)
        elif attr_method == SimplEx:
            explanation = attr_method(model, X_train, model.phi[2])
        else:
            raise ValueError(f"Invalid attribution method: {attr_method}")
        scores = metric(
            explanation,
            permutation,
            test_loader,
            device,
            N_samp=N_samp_max,
            reduce=False,
            similarity=similarity,
        )
        for n_samp in range(2, N_samp_max):
            sub_scores = scores[:, :n_samp]
            sub_scores_sem = torch.std(sub_scores) / sqrt(torch.numel(sub_scores))
            data.append(
                [
                    estimator_name,
                    n_samp,
                    torch.mean(sub_scores_sem).item(),
                    torch.mean(sub_scores).item(),
                ]
            )
    df = pd.DataFrame(
        data=data,
        columns=[
            "Estimator Name",
            "Number of MC Samples",
            "Estimator SEM",
            "Estimator Value",
        ],
    )
    df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        mc_convergence_plot(save_dir, "mnet", "mc_convergence")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_test", type=int, default=1000)
    args = parser.parse_args()
    model_name = f"dset{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_mnet_model(
            args.seed, args.latent_dim, args.batch_size, model_name=model_name
        )
    match args.name:
        case "feature_importance":
            feature_importance(
                args.seed,
                args.latent_dim,
                args.batch_size,
                args.plot,
                model_name,
                n_test=args.n_test,
            )
        case "example_importance":
            example_importance(
                args.seed,
                args.latent_dim,
                args.batch_size,
                args.plot,
                model_name,
                n_test=args.n_test,
            )
        case "concept_importance":
            concept_importance(
                args.seed,
                args.latent_dim,
                args.batch_size,
                args.plot,
                model_name,
                n_test=args.n_test,
            )
        case "mc_convergence":
            mc_convergence(
                args.seed,
                args.latent_dim,
                args.batch_size,
                args.plot,
                model_name,
                n_test=args.n_test,
            )
        case other:
            raise ValueError("Invalid experiment name.")
