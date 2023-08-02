import logging
import os
from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    GradientShap,
    IntegratedGradients,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, RandomSampler

from datasets.loaders import IMDBDataset
from interpretability.example import (
    InfluenceFunctions,
    RepresentationSimilarity,
    SimplEx,
    TracIn,
)
from interpretability.feature import FeatureImportance
from interpretability.robustness import (
    explanation_equivariance,
    explanation_invariance,
    model_invariance,
)
from models.nlp import BOWClassifier
from utils.misc import get_all_checkpoint_paths, set_random_seed
from utils.plots import single_robustness_plots
from utils.symmetries import SetPermutation


def train_model(
    random_seed: int = 42,
    data_dir: Path = Path.cwd() / "datasets/imdb",
    batch_size: int = 256,
    model_name: str = "bow_classifier",
    model_dir: Path = Path.cwd() / f"results/imdb/",
    use_wandb: bool = True,
    max_epochs: int = 20,
):
    set_random_seed(random_seed)
    datamodule = IMDBDataset(data_dir=data_dir, batch_size=batch_size)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = BOWClassifier(vocab_size=len(datamodule.token2idx))
    logger = (
        WandbLogger(project="RobustXAI", name=model_name, save_dir=model_dir)
        if use_wandb
        else None
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            every_n_epochs=1,
            save_top_k=-1,
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        default_root_dir=model_dir,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


def feature_importance(
    random_seed: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/imdb/",
    data_dir: Path = Path.cwd() / "datasets/imdb",
    plot: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    datamodule = IMDBDataset(data_dir=data_dir, batch_size=batch_size)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    permutation_group = SetPermutation()
    model = BOWClassifier.load_from_checkpoint(model_dir / "last.ckpt")
    attr_methods = {
        "DeepLift": DeepLift,
        "Integrated Gradients": IntegratedGradients,
        "Gradient Shap": GradientShap,
    }
    save_dir = model_dir / "feature_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    metrics = []
    logging.info(f"Now working with {model_name} classifier")
    model.to(device).eval()
    model_inv = model_invariance(model, permutation_group, test_loader, device)
    logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
    for attr_name, attr_method in attr_methods.items():
        logging.info(f"Now working with {attr_name} explainer")
        feat_importance = FeatureImportance(attr_method(model))
        explanation_equiv = explanation_equivariance(
            feat_importance, permutation_group, test_loader, device
        )
        for inv, equiv in zip(model_inv, explanation_equiv):
            metrics.append(
                {
                    "Model Type": model_name,
                    "Explanation": attr_name,
                    "Model Invariance": inv.item(),
                    "Explanation Equivariance": equiv.item(),
                }
            )
        logging.info(f"Explanation equivariance: {torch.mean(explanation_equiv):.3g}")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "imdb", "feature_importance")


def example_importance(
    random_seed: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/imdb/",
    data_dir: Path = Path.cwd() / "datasets/imdb",
    n_train: int = 100,
    recursion_depth: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    save_dir = model_dir / "example_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    datamodule = IMDBDataset(data_dir=data_dir, batch_size=batch_size)
    datamodule.setup("train")
    train_set = datamodule.train_set
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_loader = datamodule.predict_dataloader()
    model = BOWClassifier.load_from_checkpoint(model_dir / "last.ckpt")
    attr_methods = {
        "Representation Similarity": RepresentationSimilarity,
        "TracIn": TracIn,
        "Influence Functions": InfluenceFunctions,
        "SimplEx": SimplEx,
    }
    symmetry_group = SetPermutation()
    metrics = []
    logging.info(f"Now working with {model_name} classifier")
    model.to(device).eval()
    model_inv = model_invariance(model, symmetry_group, test_loader, device)
    logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
    model_layers = {"Embedding": model.fc2}
    for attr_name in attr_methods:
        logging.info(f"Now working with {attr_name} explainer")
        model = BOWClassifier.load_from_checkpoint(model_dir / "last.ckpt")
        if attr_name in {"TracIn", "Influence Functions"}:
            ex_importance = attr_methods[attr_name](
                model,
                X_train,
                Y_train=Y_train,
                train_loader=train_loader_replacement,
                loss_function=nn.CrossEntropyLoss(),
                save_dir=save_dir / model_name,
                recursion_depth=recursion_depth,
                checkpoint_files=get_all_checkpoint_paths(model_dir),
            )
            explanation_inv = explanation_invariance(
                ex_importance, symmetry_group, test_loader, device
            )
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append(
                    {
                        "Model Type": model_name,
                        "Explanation": attr_name,
                        "Model Invariance": inv_model.item(),
                        "Explanation Invariance": inv_expl.item(),
                    }
                )
            logging.info(f"Explanation invariance: {torch.mean(explanation_inv):.3g}")
        else:
            for layer_name in model_layers:
                ex_importance = attr_methods[attr_name](
                    model, X_train, Y_train=Y_train, layer=model_layers[layer_name]
                )
                explanation_inv = explanation_invariance(
                    ex_importance, symmetry_group, test_loader, device
                )
                ex_importance.remove_hook()
                for inv_model, inv_expl in zip(model_inv, explanation_inv):
                    metrics.append(
                        {
                            "Model Type": model_name,
                            "Explanation": f"{attr_name}-{layer_name}",
                            "Model Invariance": inv_model.item(),
                            "Explanation Invariance": inv_expl.item(),
                        }
                    )
                logging.info(
                    f"Explanation invariance for {layer_name}: {torch.mean(explanation_inv):.3g}"
                )
    metrics_df = pd.DataFrame(data=metrics)
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "imdb", "example_importance")


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("--batch_size", type=int, default=200)
@click.option("--use_wandb", is_flag=True)
@click.option("--max_epochs", type=int, default=20)
@click.option("--name", type=str, default="feature_importance")
@click.option("--plot", is_flag=True)
@click.option("--train", is_flag=True)
def main(
    seed: int,
    batch_size: int,
    use_wandb: bool,
    max_epochs: int,
    name: str,
    plot: bool,
    train: bool,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    model_name = "bow_classifier"
    if train:
        train_model(
            random_seed=seed,
            batch_size=batch_size,
            use_wandb=use_wandb,
            max_epochs=max_epochs,
            model_name=model_name,
        )

    match name:
        case "feature_importance":
            feature_importance(
                random_seed=seed,
                batch_size=batch_size,
                model_name=model_name,
                plot=plot,
            )
        case other:
            raise NotImplementedError(f"Unknown experiment name {name}.")


if __name__ == "__main__":
    main()
