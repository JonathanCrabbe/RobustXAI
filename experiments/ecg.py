import numpy as np
import torch
import torch.nn as nn
import os
import logging
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, Subset
from datasets.loaders import ECGDataset
from models.time_series import AllCNN, StandardCNN
from utils.symmetries import Translation1D
from utils.misc import set_random_seed
from utils.plots import robustness_plots, relaxing_invariance_plots
from interpretability.robustness import model_invariance, explanation_equivariance, explanation_invariance
from interpretability.example import SimplEx, RepresentationSimilarity, TracIN, InfluenceFunctions
from interpretability.feature import FeatureImportance
from captum.attr import IntegratedGradients, GradientShap, FeaturePermutation, FeatureAblation, Occlusion
from sklearn.metrics import mean_absolute_error


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
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Standard-CNN': StandardCNN(latent_dim, f'{model_name}_standard'),
              'Augmented-CNN': StandardCNN(latent_dim, f'{model_name}_augmented')}
    train_set = ECGDataset(data_dir, train=True, balance_dataset=True)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    for model_type in models:
        logging.info(f'Now fitting a {model_type} classifier')
        if model_type == 'Augmented-CNN':
            models[model_type].fit(device, train_loader, test_loader, model_dir, augmentation=True, checkpoint_interval=10)
        else:
            models[model_type].fit(device, train_loader, test_loader, model_dir, augmentation=False, checkpoint_interval=10)


def feature_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
    n_test: int = 1000
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Standard-CNN': StandardCNN(latent_dim, f'{model_name}_standard'),
              'Augmented-CNN': StandardCNN(latent_dim, f'{model_name}_augmented')}
    attr_methods = {'Integrated Gradients': IntegratedGradients, 'Gradient Shap': GradientShap,
                    'Feature Permutation': FeaturePermutation,'Feature Ablation': FeatureAblation,
                    'Feature Occlusion': Occlusion}
    model_dir = model_dir/model_name
    save_dir = model_dir/'feature_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    translation = Translation1D()
    metrics = []
    for model_type in models:
        logging.info(f'Now working with {model_type} classifier')
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(model, translation, test_loader, device, N_samp=50)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            feat_importance = FeatureImportance(attr_methods[attr_name](model))
            explanation_equiv = explanation_equivariance(feat_importance, translation, test_loader, device, N_samp=1)
            for inv, equiv in zip(model_inv, explanation_equiv):
                metrics.append([model_type, attr_name, inv.item(), equiv.item()])
            logging.info(f'Explanation equivariance: {torch.mean(explanation_equiv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Equivariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        robustness_plots(save_dir, 'ecg', 'feature_importance')
        relaxing_invariance_plots(save_dir, 'ecg', 'feature_importance')


def example_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
    n_test: int = 1000,
    n_train: int = 100,
    recursion_depth: int = 100
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=recursion_depth*batch_size)
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Standard-CNN': StandardCNN(latent_dim, f'{model_name}_standard'),
              'Augmented-CNN': StandardCNN(latent_dim, f'{model_name}_augmented')}
    attr_methods = {'Influence Functions': InfluenceFunctions, 'TracIn': TracIN, 'SimplEx': SimplEx,
                    'Representation Similarity': RepresentationSimilarity}
    model_dir = model_dir/model_name
    save_dir = model_dir/'example_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    translation = Translation1D()
    metrics = []
    for model_type in models:
        logging.info(f'Now working with {model_type} classifier')
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(model, translation, test_loader, device, N_samp=50)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            ex_importance = attr_methods[attr_name](model, X_train, Y_train=Y_train, train_loader=train_loader_replacement,
                                                    loss_function=nn.CrossEntropyLoss(), batch_size=batch_size,
                                                    save_dir=save_dir /'influence_functions'/model.name, recursion_depth=recursion_depth)
            explanation_inv = explanation_invariance(ex_importance, translation, test_loader, device, N_samp=1)
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append([model_type, attr_name, inv_model.item(), inv_expl.item()])
            logging.info(f'Explanation invariance: {torch.mean(explanation_inv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        robustness_plots(save_dir, 'ecg', 'example_importance')
        relaxing_invariance_plots(save_dir, 'ecg', 'example_importance')


def understand_randomness(
        random_seed: int,
        latent_dim: int,
        batch_size: int,
        plot: bool,
        model_name: str,
        model_dir: Path = Path.cwd() / f"results/ecg/",
        data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Random-CNN': StandardCNN(latent_dim)}
    model_dir = model_dir / model_name
    save_dir = model_dir / 'understand_randomness'
    if not save_dir.exists():
        os.makedirs(save_dir)
    model = AllCNN(latent_dim).to(device)
    model.load_state_dict(torch.load(model_dir / f"{model_name}_allcnn.pt"), strict=False)
    for model_type in models:
        predictions = []
        model = models[model_type]
        if model_type != 'Random-CNN':
            model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device)
        model.eval()
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            predictions.append(model(x).detach().cpu().numpy())
            T = x.shape[-1]
        predictions = np.concatenate(predictions)
        baseline = model(torch.zeros((1, 1, T), device=device)).detach().cpu().numpy()  # Record baseline prediction
        baseline = np.tile(baseline, [len(predictions), 1])
        logging.info(f'{model_type}: {mean_absolute_error(predictions, baseline):.3g}')


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
        train_ecg_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
        case 'example_importance':
            example_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
        case 'understand_randomness':
            understand_randomness(args.seed, args.latent_dim, args.batch_size, args.plot, model_name)
        case other:
            raise ValueError('Unrecognized experiment name.')
