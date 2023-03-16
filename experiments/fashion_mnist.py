import torch
import os
import logging
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from models.images import AllCNN
from pathlib import Path
from utils.misc import set_random_seed
from utils.plots import single_robustness_plots
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    FeaturePermutation,
    FeatureAblation,
    Occlusion,
)
from utils.symmetries import Translation2D
from interpretability.robustness import (
    model_invariance_exact,
    explanation_equivariance_exact,
)
from interpretability.feature import FeatureImportance
from torch.utils.data import Subset


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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Pad(max_displacement)]
    )
    train_set = FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = FashionMNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    model = AllCNN(latent_dim)
    model.fit(
        device,
        train_loader,
        test_loader,
        model_dir,
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
    test_size: int = 500,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Pad(max_displacement)]
    )
    test_set = FashionMNIST(data_dir, train=False, transform=transform, download=True)
    small_test_set = Subset(test_set, torch.randperm(len(test_set))[:test_size])
    test_loader = DataLoader(small_test_set, batch_size=100, shuffle=False)
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
            metrics.append(["GNN", attr_name, inv.item(), equiv.item()])
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
    args = parser.parse_args()
    model_name = f"cnn{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_fashion_mnist_model(
            args.seed, args.latent_dim, args.batch_size, model_name=model_name
        )
    match args.name:
        case "feature_importance":
            feature_importance(args.seed, args.latent_dim, args.plot, model_name)
        case other:
            raise ValueError("Invalid experiment name.")
