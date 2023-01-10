import logging
import argparse
import torch
import os
from datasets.loaders import ModelNet40Dataset
from pathlib import Path
from utils.misc import set_random_seed
from models.sets import ClassifierModelNet40
from torch.utils.data import DataLoader

def train_mnet_model(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mnet/",
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
    model.fit(device, train_loader, test_loader, model_dir,  checkpoint_interval=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_test", type=int, default=1000)
    args = parser.parse_args()
    model_name = f"dset{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_mnet_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
