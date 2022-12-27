import torch
import os
import logging
import argparse
from torch_geometric.loader import DataLoader
from datasets.loaders import MutagenicityDataset
from models.graphs import ClassifierMutagenicity
from pathlib import Path
from utils.misc import set_random_seed
from captum.attr import IntegratedGradients, GradientShap, FeaturePermutation, FeatureAblation
from utils.symmetries import GraphPermutation


def train_mut_model(random_seed: int, latent_dim: int, batch_size: int, model_name: str = "model",
                    model_dir: Path = Path.cwd() / f"results/mut/", data_dir: Path = Path.cwd() / "datasets/mut") -> None:
    logging.info("Fitting the Mutagenicity classifiers")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_set = MutagenicityDataset(data_dir, train=True)
    test_set = MutagenicityDataset(data_dir, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    model = ClassifierMutagenicity(latent_dim)
    model.fit(device, train_loader, test_loader, model_dir, checkpoint_interval=20)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mut/",
    data_dir: Path = Path.cwd() / "datasets/mut",
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = MutagenicityDataset(data_dir, train=False)
    test_loader = DataLoader(test_set, batch_size=1)
    model_dir = model_dir / model_name
    model = ClassifierMutagenicity(latent_dim)
    model.load_metadata(model_dir)
    model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
    attr_methods = {'Integrated Gradients': IntegratedGradients, 'Gradient Shap': GradientShap,
                    'Feature Permutation': FeaturePermutation, 'Feature Ablation': FeatureAblation}
    save_dir = model_dir/'feature_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    graph_perm = GraphPermutation()
    metrics = []
    for data in test_loader:
        data = data.to(device)
        new_data = graph_perm(data)
        print(data.x)
        print(new_data.x)
        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_test", type=int, default=1000)
    args = parser.parse_args()
    model_name = f"cnn{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_mut_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.seed, args.latent_dim, args.plot, model_name)
