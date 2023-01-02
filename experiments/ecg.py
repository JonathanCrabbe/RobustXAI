import numpy as np
import torch
import torch.nn as nn
import os
import logging
import argparse
import pandas as pd
import itertools
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, Subset
from datasets.loaders import ECGDataset
from models.time_series import AllCNN, StandardCNN
from utils.symmetries import Translation1D
from utils.misc import set_random_seed
from utils.plots import single_robustness_plots, relaxing_invariance_plots, mc_convergence_plot, enforce_invariance_plot
from interpretability.robustness import model_invariance, explanation_equivariance, explanation_invariance, \
    accuracy, cos_similarity, InvariantExplainer, model_invariance_exact, explanation_invariance_exact, \
    explanation_equivariance_exact
from interpretability.example import SimplEx, RepresentationSimilarity, TracIn, InfluenceFunctions
from interpretability.feature import FeatureImportance
from interpretability.concept import CAR, CAV, ConceptExplainer
from captum.attr import IntegratedGradients, GradientShap, FeaturePermutation, FeatureAblation, Occlusion
from sklearn.metrics import mean_absolute_error
from math import sqrt


def train_ecg_model(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
) -> None:
    logging.info("Fitting the ECG classifiers")
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
    n_test: int = 1000,
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
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            feat_importance = FeatureImportance(attr_methods[attr_name](model))
            explanation_equiv = explanation_equivariance_exact(feat_importance, translation, test_loader, device)
            for inv, equiv in zip(model_inv, explanation_equiv):
                metrics.append([model_type, attr_name, inv.item(), equiv.item()])
            logging.info(f'Explanation equivariance: {torch.mean(explanation_equiv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Equivariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        single_robustness_plots(save_dir, 'ecg', 'feature_importance')
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
    recursion_depth: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ECGDataset(data_dir, train=True, balance_dataset=False)
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
    attr_methods = {'SimplEx': SimplEx, 'Representation Similarity': RepresentationSimilarity,
                    'TracIn': TracIn, 'Influence Functions': InfluenceFunctions, }
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
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        model_layers = {'Lin1': model.fc1, 'Conv3': model.cnn3}
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
            if attr_name in {'TracIn', 'Influence Functions'}:
                ex_importance = attr_methods[attr_name](model, X_train, Y_train=Y_train, train_loader=train_loader_replacement,
                                                        loss_function=nn.CrossEntropyLoss(), save_dir=save_dir/model.name,
                                                        recursion_depth=recursion_depth,)
                explanation_inv = explanation_invariance_exact(ex_importance, translation, test_loader, device)
                for inv_model, inv_expl in zip(model_inv, explanation_inv):
                    metrics.append([model_type, attr_name, inv_model.item(), inv_expl.item()])
                logging.info(f'Explanation invariance: {torch.mean(explanation_inv):.3g}')
            else:
                for layer_name in model_layers:
                    ex_importance = attr_methods[attr_name](model, X_train, Y_train=Y_train, layer=model_layers[layer_name])
                    explanation_inv = explanation_invariance_exact(ex_importance, translation, test_loader, device)
                    ex_importance.remove_hook()
                    for inv_model, inv_expl in zip(model_inv, explanation_inv):
                        metrics.append([model_type, f'{attr_name}-{layer_name}', inv_model.item(), inv_expl.item()])
                    logging.info(f'Explanation invariance for {layer_name}: {torch.mean(explanation_inv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        single_robustness_plots(save_dir, 'ecg', 'example_importance')
        relaxing_invariance_plots(save_dir, 'ecg', 'example_importance')


def concept_importance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
    n_test: int = 1000,
    concept_set_size: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ECGDataset(data_dir, train=True, binarize_label=False, balance_dataset=False)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False, binarize_label=False)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn'),
              'Standard-CNN': StandardCNN(latent_dim, f'{model_name}_standard'),
              'Augmented-CNN': StandardCNN(latent_dim, f'{model_name}_augmented')}
    attr_methods = {'CAV': CAV, 'CAR': CAR}
    model_dir = model_dir/model_name
    save_dir = model_dir/'concept_importance'
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
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        model_layers = {'Lin1': model.fc1, 'Conv3': model.cnn3}
        for layer_name, attr_name in itertools.product(model_layers, attr_methods):
            logging.info(f'Now working with {attr_name} explainer on layer {layer_name}')
            conc_importance = attr_methods[attr_name](model, train_set, n_classes=2, layer=model_layers[layer_name])
            conc_importance.fit(device, concept_set_size)
            concept_acc = conc_importance.concept_accuracy(test_set, device, concept_set_size=concept_set_size)
            for concept_name in concept_acc:
                logging.info(f'Concept {concept_name} accuracy: {concept_acc[concept_name]:.2g}')
            explanation_inv = explanation_invariance_exact(conc_importance, translation, test_loader, device, similarity=accuracy)
            conc_importance.remove_hook()
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append([model_type, f'{attr_name}-{layer_name}', inv_model.item(), inv_expl.item()])
            logging.info(f'Explanation invariance: {torch.mean(explanation_inv):.3g}')
    metrics_df = pd.DataFrame(data=metrics, columns=['Model Type', 'Explanation', 'Model Invariance', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir/'metrics.csv', index=False)
    if plot:
        single_robustness_plots(save_dir, 'ecg', 'concept_importance')
        relaxing_invariance_plots(save_dir, 'ecg', 'concept_importance')


def enforce_invariance(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
    n_test: int = 1000,
    concept_set_size: int = 100,
    N_samp: int = 50
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ECGDataset(data_dir, train=True, binarize_label=False, balance_dataset=False)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    models = {'All-CNN': AllCNN(latent_dim, f'{model_name}_allcnn')}
    attr_methods = {'CAV': CAV, 'CAR': CAR}
    model_dir = model_dir/model_name
    save_dir = model_dir/'enforce_invariance'
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
        model_inv = model_invariance_exact(model, translation, test_loader, device)
        logging.info(f'Model invariance: {torch.mean(model_inv):.3g}')
        for attr_name in attr_methods:
            logging.info(f'Now working with {attr_name} explainer')
            attr_method = attr_methods[attr_name](model, train_set, n_classes=2)
            if isinstance(attr_method, ConceptExplainer):
                attr_method.fit(device, concept_set_size)
            for N_inv in [1, 5, 20, 50, 100, 150, 187]:
                logging.info(f'Now working with invariant explainer with N_inv = {N_inv}')
                inv_method = InvariantExplainer(attr_method, translation, N_inv, isinstance(attr_method, ConceptExplainer))
                explanation_inv = explanation_invariance_exact(inv_method, translation, test_loader, device, similarity=accuracy)
                logging.info(f'N_inv = {N_inv} - Explanation invariance = {torch.mean(explanation_inv):.3g}')
                for inv_expl in explanation_inv:
                    metrics.append([model_type, attr_name, N_inv, inv_expl.item()])
    metrics_df = pd.DataFrame(data=metrics,
                              columns=['Model Type', 'Explanation', 'N_inv', 'Explanation Invariance'])
    metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
    if plot:
        enforce_invariance_plot(save_dir, 'ecg')


def mc_convergence(
    random_seed: int,
    latent_dim: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/ecg/",
    data_dir: Path = Path.cwd() / "datasets/ecg",
    n_train: int = 100,
    n_test: int = 1000,
    N_samp_max: int = 100
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    train_set = ECGDataset(data_dir, train=True, balance_dataset=False, binarize_label=False)
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, _ = next(iter(train_loader))
    X_train = X_train.to(device)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    test_subset = Subset(test_set, torch.randperm(len(test_set))[:n_test])
    test_loader = DataLoader(test_subset, batch_size)
    model_dir = model_dir / model_name
    model = StandardCNN(latent_dim, f'{model_name}_augmented')
    model.load_metadata(model_dir)
    model.load_state_dict(torch.load(model_dir / f"{model.name}.pt"), strict=False)
    model.to(device).eval()
    translation = Translation1D()
    save_dir = model_dir/'mc_convergence'
    mc_estimators = {
        'Augmented-CNN Invariance': (model, model_invariance, cos_similarity),
        'Integrated Gradients Equivariance': (FeatureImportance(IntegratedGradients(model)), explanation_equivariance, cos_similarity),
        'Representation Similarity Invariance': (RepresentationSimilarity(model, X_train), explanation_invariance, cos_similarity),
        'CAV Invariance': (CAV(model, train_set, 2), explanation_invariance, accuracy)
        }
    if not save_dir.exists():
        os.makedirs(save_dir)
    data = []
    for estimator_name in mc_estimators:
        logging.info(f'Computing Monte Carlo estimators for {estimator_name}')
        function, metric, similarity = mc_estimators[estimator_name]
        if isinstance(function, ConceptExplainer):
            function.fit(device, 100)
        scores = metric(function, translation, test_loader, device, N_samp=N_samp_max, reduce=False, similarity=similarity)
        for n_samp in range(2, N_samp_max):
            sub_scores = scores[:, :n_samp]
            sub_scores_sem = torch.std(sub_scores, dim=-1)/sqrt(n_samp)
            data.append([estimator_name, n_samp, torch.mean(sub_scores_sem).item(), torch.mean(sub_scores).item()])
    df = pd.DataFrame(data=data, columns=['Estimator Name', 'Number of MC Samples', 'Estimator SEM', 'Estimator Value'])
    df.to_csv(save_dir / 'metrics.csv', index=False)
    if plot:
        mc_convergence_plot(save_dir, 'ecg', 'mc_convergence')


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
    parser.add_argument("--batch_size", type=int, default=500)
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
        case 'concept_importance':
            concept_importance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
        case 'enforce_invariance':
            enforce_invariance(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
        case 'mc_convergence':
            mc_convergence(args.seed, args.latent_dim, args.batch_size, args.plot, model_name, n_test=args.n_test)
        case 'understand_randomness':
            understand_randomness(args.seed, args.latent_dim, args.batch_size, args.plot, model_name)
        case other:
            raise ValueError('Unrecognized experiment name.')
