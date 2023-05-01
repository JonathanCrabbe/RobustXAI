import torch
import os
import logging
import argparse
import pandas as pd
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models.images import AllCNN, StandardCNN
from pathlib import Path
from datasets.loaders import FashionMnistDataset
from utils.misc import set_random_seed
from utils.plots import single_robustness_plots, relaxing_invariance_plots
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    FeaturePermutation,
    FeatureAblation,
    Occlusion,
)
from interpretability.example import (
    SimplEx,
    InfluenceFunctions,
    RepresentationSimilarity,
    TracIn,
)
from interpretability.concept import CAR, CAV
from utils.symmetries import Translation2D
from interpretability.robustness import (
    model_invariance_exact,
    explanation_equivariance_exact,
    explanation_invariance_exact,
    accuracy,
)
from interpretability.feature import FeatureImportance
from torch.utils.data import Subset, RandomSampler


def train_fashion_mnist_model(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/fashion_mnist/",
    data_dir: Path = Path.cwd() / "datasets/fashion_mnist",
    max_displacement: int = 10,
) -> None:
    logging.info("Fitting the Fashion-Mnist classifiers")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    train_set = FashionMnistDataset(
        data_dir, train=True, max_displacement=max_displacement
    )
    test_set = FashionMnistDataset(
        data_dir, train=False, max_displacement=max_displacement
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    models = {
        "Augmented-CNN": StandardCNN(latent_dim, f"{model_name}_augmented"),
        "All-CNN": AllCNN(latent_dim, f"{model_name}_allcnn"),
        "Standard-CNN": StandardCNN(latent_dim, f"{model_name}_standard"),
    }
    for model_type in models:
        logging.info(f"Now fitting a {model_type} classifier")
        if model_type == "Augmented-CNN":
            models[model_type].fit(
                device,
                train_loader,
                test_loader,
                model_dir,
                augmentation=True,
                checkpoint_interval=20,
                patience=50,
                n_epoch=500,
            )
        else:
            models[model_type].fit(
                device,
                train_loader,
                test_loader,
                model_dir,
                augmentation=False,
                checkpoint_interval=20,
                patience=50,
                n_epoch=500,
            )


def feature_importance(
    random_seed: int,
    latent_dim: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/fashion_mnist/",
    data_dir: Path = Path.cwd() / "datasets/fashion_mnist",
    max_displacement: int = 10,
    n_test: int = 500,
    batch_size: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = FashionMnistDataset(
        data_dir, train=False, max_displacement=max_displacement
    )
    small_test_set = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(small_test_set, batch_size=batch_size, shuffle=False)
    model_dir = model_dir / model_name
    model = AllCNN(latent_dim)
    model.load_metadata(model_dir)
    model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
    model.to(device).eval()
    attr_methods = {
        "Integrated Gradients": IntegratedGradients,
        "Gradient Shap": GradientShap,
        "Feature Permutation": FeaturePermutation,
        "Feature Ablation": FeatureAblation,
        "Feature Occlusion": Occlusion,
    }
    save_dir = model_dir / "feature_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    translation = Translation2D(max_dispacement=max_displacement)
    metrics = []
    logging.info(f"Now working with Fashion Mnist classifier")
    model_inv = model_invariance_exact(model, translation, test_loader, device)
    logging.info(f"Model invariance: {torch.mean(model_inv).item():.3g}")
    for attr_name in attr_methods:
        logging.info(f"Now working with {attr_name}")
        feat_importance = FeatureImportance(attr_methods[attr_name](model))
        explanation_equiv = explanation_equivariance_exact(
            feat_importance, translation, test_loader, device
        )
        logging.info(f"Explanation equivariance: {torch.mean(explanation_equiv):.3g}")
        for inv, equiv in zip(model_inv, explanation_equiv):
            metrics.append(["CNN", attr_name, inv.item(), equiv.item()])
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
        single_robustness_plots(save_dir, "fashion_mnist", "feature_importance")


def example_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/fashion_mnist/",
    data_dir: Path = Path.cwd() / "datasets/fashion_mnist",
    n_test: int = 1000,
    n_train: int = 100,
    recursion_depth: int = 100,
    max_displacement: int = 10,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = FashionMnistDataset(
        data_dir, train=True, max_displacement=max_displacement
    )
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_set = FashionMnistDataset(
        data_dir, train=False, max_displacement=max_displacement
    )
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {
        "All-CNN": AllCNN(latent_dim, f"{model_name}_allcnn"),
        "Standard-CNN": StandardCNN(latent_dim, f"{model_name}_standard"),
        "Augmented-CNN": StandardCNN(latent_dim, f"{model_name}_augmented"),
    }
    attr_methods = {
        "SimplEx": SimplEx,
        "Representation Similarity": RepresentationSimilarity,
        "TracIn": TracIn,
        "Influence Functions": InfluenceFunctions,
    }
    model_dir = model_dir / model_name
    save_dir = model_dir / "example_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    translation = Translation2D(max_displacement)
    metrics = []
    for model_type in models:
        logging.info(f"Now working with {model_type} classifier")
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
        model_layers = {"Lin1": model.fc1, "Conv3": model.cnn3}
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
                explanation_inv = explanation_invariance_exact(
                    ex_importance, translation, test_loader, device
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
                    explanation_inv = explanation_invariance_exact(
                        ex_importance, translation, test_loader, device
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
        single_robustness_plots(save_dir, "fashion_mnist", "example_importance")
        relaxing_invariance_plots(save_dir, "fashion_mnist", "example_importance")


def concept_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/fashion_mnist/",
    data_dir: Path = Path.cwd() / "datasets/fashion_mnist",
    n_test: int = 1000,
    concept_set_size: int = 100,
    max_displacement: int = 10,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = FashionMnistDataset(
        data_dir, train=True, max_displacement=max_displacement
    )
    test_set = FashionMnistDataset(
        data_dir, train=False, max_displacement=max_displacement
    )
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {
        "All-CNN": AllCNN(latent_dim, f"{model_name}_allcnn"),
        "Standard-CNN": StandardCNN(latent_dim, f"{model_name}_standard"),
        "Augmented-CNN": StandardCNN(latent_dim, f"{model_name}_augmented"),
    }
    attr_methods = {"CAV": CAV, "CAR": CAR}
    model_dir = model_dir / model_name
    save_dir = model_dir / "concept_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    translation = Translation2D(max_displacement)
    metrics = []
    for model_type in models:
        logging.info(f"Now working with {model_type} classifier")
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
        model_layers = {"Lin1": model.fc1, "Conv3": model.cnn3}
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
            explanation_inv = explanation_invariance_exact(
                conc_importance, translation, test_loader, device, similarity=accuracy
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
        single_robustness_plots(save_dir, "ecg", "concept_importance")
        relaxing_invariance_plots(save_dir, "ecg", "concept_importance")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_test", type=int, default=500)
    args = parser.parse_args()
    model_name = f"cnn{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_fashion_mnist_model(
            args.seed, args.latent_dim, args.batch_size, model_name=model_name
        )
    match args.name:
        case "feature_importance":
            feature_importance(
                args.seed,
                args.latent_dim,
                args.plot,
                model_name,
                batch_size=args.batch_size,
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
        case other:
            raise ValueError("Invalid experiment name.")
