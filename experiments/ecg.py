import torch
import os
import logging
import numpy as np
import argparse
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.loaders import ECGDataset
from models.time_series import AllCNN, StandardCNN
from utils.symmetries import Translation1D
from utils.metrics import AverageMeter
from utils.misc import set_random_seed
from tqdm import tqdm
from interpretability.robustness import invariance

concept_to_class = {
    "Supraventricular": 1,
    "Premature Ventricular": 2,
    "Fusion Beats": 3,
    "Unknown": 4,
}


def train_ecg_model(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:
    logging.info("Fitting the ECG classifiers.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    models = [AllCNN(latent_dim, f'{model_name}_allcnn'), StandardCNN(latent_dim, f'{model_name}_standard')]
    train_set = ECGDataset(data_dir, train=True, balance_dataset=True)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    for model in models:
        logging.info(f'Now fitting a {model.name} classifier.')
        model.fit(device, train_loader, test_loader, model_dir)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    models = [AllCNN(latent_dim, f'{model_name}_allcnn'), StandardCNN(latent_dim, f'{model_name}_standard')]
    model_dir = model_dir/model_name
    for model in models:
        model.load_state_dict(torch.load(model_dir/ f"{model.name}.pt"), strict=False)
        model.to(device)
        model.eval()
        translation = Translation1D()
        invariance_scores = invariance(model, translation, test_loader, device)
        print(torch.mean(invariance_scores))
        sns.histplot(invariance_scores)
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    model_name = f"cnn{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_ecg_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.seed, args.latent_dim, args.batch_size, model_name)
        case other:
            logging.info('Unrecognized experiment name.')
