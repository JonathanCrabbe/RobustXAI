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
from models.time_series import AllCNNECG, ClassifierECG
from utils.symmetries import Translation1D
from utils.metrics import AverageMeter
from tqdm import tqdm

concept_to_class = {
    "Supraventricular": 1,
    "Premature Ventricular": 2,
    "Fusion Beats": 3,
    "Unknown": 4,
}


def train_ecg_model(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:
    logging.info("Fitting an ECG Classifier")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ClassifierECG(latent_dim, model_name).to(device) #AllCNNECG(model_name)
    train_set = ECGDataset(data_dir, train=True, balance_dataset=True)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    model.fit(device, train_loader, test_loader, model_dir)


def feature_importance(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ClassifierECG(latent_dim, model_name).to(device) # AllCNNECG(model_name).to(device)
    model_dir = model_dir/model_name
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.eval()

    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    transl = Translation1D(12)
    mae = AverageMeter('mae')
    maes = np.empty(0)
    switch = AverageMeter('switch')
    for x, _ in test_loader:
        x = x.to(device)
        x_transl = transl(x)
        #print(x[0])
        #print(x_transl[0])
        p1 = F.softmax(model(x), -1)
        c1 = torch.amax(p1, -1)
        p2 = F.softmax(model(x_transl), -1)
        c2 = torch.amax(p2, -1)
        maes = np.concatenate((maes, torch.sum(torch.abs(p1-p2), dim=-1).detach().cpu().numpy()))
        mae.update(torch.mean(torch.abs(p1-p2)), len(x))
        switch.update(torch.mean(torch.where(c1 == c2, 0., 1.)), len(x))
    print(mae.avg)
    print(np.percentile(maes, 50))
    print(switch.avg)
    sns.histplot(maes)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    model_name = f"cnn{args.latent_dim}"
    if args.train:
        train_ecg_model(args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.latent_dim, args.batch_size, model_name)
        case other:
            logging.info('Unrecognized experiment name.')
