import torch
import os
import logging
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.loaders import ECGDataset
from models.time_series import AllCNN, StandardCNN
from utils.symmetries import Translation1D
from utils.misc import set_random_seed
from utils.plots import robustness_plots, relaxing_invariance_plots
from itertools import product
from interpretability.robustness import invariance, equivariance
from interpretability.feature import FeatureImportance
from captum.attr import IntegratedGradients, GradientShap, FeaturePermutation, FeatureAblation, Saliency


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
        logging.info(f'Now fitting a {model.name} classifier')
        model.fit(device, train_loader, test_loader, model_dir)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    models = {'All Convolutional': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Standard': StandardCNN(latent_dim, f'{model_name}_standard')}
    attr_methods = {'Integrated Gradients': IntegratedGradients,
                    'GradientShap': GradientShap,
                    'Saliency': Saliency,
                    'Feature Permutation': FeaturePermutation,
                    'Feature Ablation': FeatureAblation
                    }
    model_dir = model_dir/model_name
    translation = Translation1D()
    metrics = []
    for model_type, attr_name in product(models, attr_methods):
        logging.info(f'Now working with classifier = {model_type} and explainer = {attr_name}')
        model = models[model_type]
        model.load_state_dict(torch.load(model_dir/f"{model.name}.pt"), strict=False)
        model.to(device)
        model.eval()
        feat_importance = FeatureImportance(attr_methods[attr_name](model))
        model_invariance = invariance(model, translation, test_loader, device, N_samp=1)
        explanation_equivariance = equivariance(feat_importance, translation, test_loader, device, N_samp=1)
        for inv, equiv in zip(model_invariance, explanation_equivariance):
            metrics.append([model_type, attr_name, inv.item(), equiv.item()])
        logging.info(f'Model invariance: {torch.mean(model_invariance):.3g}')
        logging.info(f'Explanation equivariance: {torch.mean(explanation_equivariance):.3g}')
    metrics_df = pd.DataFrame(data=metrics,
                              columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Equivariance'])
    metrics_df.to_csv(model_dir/'metrics.csv', index=False)
    if plot:
        robustness_plots(model_dir, 'ecg')
        relaxing_invariance_plots(model_dir, 'ecg')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
            feature_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name)
        case other:
            raise ValueError('Unrecognized experiment name.')
