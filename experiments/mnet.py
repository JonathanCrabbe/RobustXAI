import logging
import argparse
import torch
import os
import pandas as pd
from datasets.loaders import ModelNet40Dataset
from pathlib import Path
from utils.misc import set_random_seed
from models.sets import ClassifierModelNet40
from torch.utils.data import DataLoader, Subset
from captum.attr import IntegratedGradients, GradientShap, FeatureAblation, FeaturePermutation
from utils.symmetries import SetPermutation
from interpretability.feature import FeatureImportance
from interpretability.robustness import model_invariance, explanation_equivariance
from utils.plots import single_robustness_plots



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


def feature_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mnet/",
    data_dir: Path = Path.cwd() / "datasets/mnet",
    n_test: int = 1000,
    N_samp: int = 50
) -> None:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    test_set = ModelNet40Dataset(data_dir, train=False, random_seed=random_seed)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {'Deep-Set': ClassifierModelNet40(latent_dim=latent_dim, name=model_name)}
    attr_methods = {'Gradient Shap': GradientShap, 'Integrated Gradients': IntegratedGradients,
                    'Feature Permutation': FeaturePermutation, 'Feature Ablation': FeatureAblation,}
    model_dir = model_dir/model_name
    save_dir = model_dir/'feature_importance'
    if not save_dir.exists():
        os.makedirs(save_dir)
    permutation = SetPermutation()
    metrics = []
    for model_type in models:
        logging.info(f'Now working with {model_type} classifier')
        model = models[model_type]
        model.load_metadata(model_dir)
        model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
        model.to(device).eval()
        model_inv = model_invariance(model, permutation, test_loader, device, N_samp=N_samp)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            feat_importance = FeatureImportance(attr_methods[attr_name](model))
            explanation_equiv = explanation_equivariance(feat_importance, permutation, test_loader, device, N_samp=N_samp)
            for inv, equiv in zip(model_inv, explanation_equiv):
                metrics.append([model_type, attr_name, inv.item(), equiv.item()])
            logging.info(f'Explanation equivariance: {torch.mean(explanation_equiv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Equivariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        single_robustness_plots(save_dir, 'mnet', 'feature_importance')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_test", type=int, default=1000)
    args = parser.parse_args()
    model_name = f"dset{args.latent_dim}_seed{args.seed}"
    if args.train:
        train_mnet_model(args.seed, args.latent_dim, args.batch_size, model_name=model_name)
    match args.name:
        case 'feature_importance':
            feature_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
