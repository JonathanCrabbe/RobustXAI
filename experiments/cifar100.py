import pytorch_lightning as pl
import argparse
import torch
import logging
import os
import pandas as pd
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models.images import Wide_ResNet
from datasets.loaders import Cifar100Dataset
from pathlib import Path
from utils.misc import set_random_seed
from utils.symmetries import Dihedral
from captum.attr import (
    DeepLift,
    IntegratedGradients,
    GradientShap,
)
from interpretability.feature import FeatureImportance
from interpretability.robustness import (
    model_invariance_exact,
    explanation_equivariance_exact,
    explanation_invariance_exact,
    ComputeModelInvariance,
    ComputeSaliencyEquivariance,
)
from interpretability.example import (
    TracIn,
    SimplEx,
    InfluenceFunctions,
    RepresentationSimilarity,
)
from utils.plots import single_robustness_plots
from utils.misc import get_best_checkpoint, get_all_checkpoint_paths
from torch.utils.data import DataLoader, RandomSampler


def train_cifar100_model(
    random_seed: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/cifar100/",
    data_dir: Path = Path.cwd() / "datasets/cifar100",
    use_wandb: bool = False,
    max_epochs: int = 200,
) -> None:
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = Wide_ResNet()
    datamodule = Cifar100Dataset(
        data_dir=data_dir, batch_size=batch_size, num_predict=50
    )
    logger = (
        pl.loggers.WandbLogger(project="RobustXAI", name=model_name, save_dir=model_dir)
        if use_wandb
        else None
    )
    callbacks = [
        ComputeModelInvariance(symmetry=Dihedral(), datamodule=datamodule),
        ComputeSaliencyEquivariance(symmetry=Dihedral(), datamodule=datamodule),
        ModelCheckpoint(
            dirpath=model_dir,
            monitor="val_acc",
            every_n_epochs=1,
            save_top_k=1,
            mode="max",
            filename=model_name + "-{epoch:02d}-{val_acc:.2f}",
        ),
        ModelCheckpoint(
            dirpath=model_dir,
            monitor="val_acc",
            every_n_epochs=20,
            save_top_k=-1,
            filename=model_name + "-{epoch:02d}-{val_acc:.2f}",
        ),
        EarlyStopping(monitor="val_acc", patience=20, mode="max"),
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
    model_dir: Path = Path.cwd() / f"results/cifar100/",
    data_dir: Path = Path.cwd() / "datasets/cifar100",
    plot: bool = True,
    n_test: int = 500,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    datamodule = Cifar100Dataset(
        data_dir=data_dir, batch_size=batch_size, num_predict=n_test
    )
    datamodule.setup("predict")
    test_loader = datamodule.predict_dataloader()
    dihedral_group = Dihedral()
    ckpt = torch.load(get_best_checkpoint(model_dir))
    model = Wide_ResNet()
    model_type = "D8-Wide-ResNet"
    model.load_state_dict(ckpt["state_dict"], strict=False)
    attr_methods = {
        "DeepLift": DeepLift,
        "Integrated Gradients": IntegratedGradients,
        "Gradient Shap": GradientShap,
    }
    save_dir = model_dir / "feature_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    metrics = []
    logging.info(f"Now working with {model_type} classifier")
    model.to(device).eval()
    model_inv = model_invariance_exact(model, dihedral_group, test_loader, device)
    logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
    for attr_name, attr_method in attr_methods.items():
        logging.info(f"Now working with {attr_name} explainer")
        feat_importance = FeatureImportance(attr_method(model))
        explanation_equiv = explanation_equivariance_exact(
            feat_importance, dihedral_group, test_loader, device
        )
        for inv, equiv in zip(model_inv, explanation_equiv):
            metrics.append(
                {
                    "Model Type": model_type,
                    "Explanation": attr_name,
                    "Model Invariance": inv.item(),
                    "Explanation Equivariance": equiv.item(),
                }
            )
        logging.info(f"Explanation equivariance: {torch.mean(explanation_equiv):.3g}")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        single_robustness_plots(save_dir, "cifar100", "feature_importance")


def example_importance(
    random_seed: int,
    batch_size: int,
    plot: bool,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/cifar100/",
    data_dir: Path = Path.cwd() / "datasets/cifar100",
    n_test: int = 1000,
    n_train: int = 100,
    recursion_depth: int = 100,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    save_dir = model_dir / "example_importance"
    if not save_dir.exists():
        os.makedirs(save_dir)
    datamodule = Cifar100Dataset(
        data_dir=data_dir, batch_size=batch_size, num_predict=n_test
    )
    datamodule.setup("predict")
    train_set = datamodule.cifar100_train
    train_loader = DataLoader(train_set, n_train, shuffle=True)
    X_train, Y_train = next(iter(train_loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(train_set, batch_size, sampler=train_sampler)
    test_loader = datamodule.predict_dataloader()
    checkpoint = torch.load(get_best_checkpoint(model_dir))
    model = Wide_ResNet()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model_type = "D8-Wide-ResNet"
    attr_methods = {
        "Representation Similarity": RepresentationSimilarity,
        "TracIn": TracIn,
        "Influence Functions": InfluenceFunctions,
        "SimplEx": SimplEx,
    }
    dihedral_group = Dihedral()
    metrics = []
    logging.info(f"Now working with {model_type} classifier")
    model.to(device).eval()
    model_inv = model_invariance_exact(model, dihedral_group, test_loader, device)
    logging.info(f"Model invariance: {torch.mean(model_inv):.3g}")
    model_layers = {"Conv1": model.conv1, "Layer3": model.relu}
    for attr_name in attr_methods:
        logging.info(f"Now working with {attr_name} explainer")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
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
            explanation_inv = explanation_invariance_exact(
                ex_importance, dihedral_group, test_loader, device
            )
            for inv_model, inv_expl in zip(model_inv, explanation_inv):
                metrics.append(
                    {
                        "Model Type": model_type,
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
                explanation_inv = explanation_invariance_exact(
                    ex_importance, dihedral_group, test_loader, device
                )
                ex_importance.remove_hook()
                for inv_model, inv_expl in zip(model_inv, explanation_inv):
                    metrics.append(
                        {
                            "Model Type": model_type,
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
        single_robustness_plots(save_dir, "cifar100", "example_importance")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--name", type=str, default="feature importance")
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()
    model_name = f"cifar100_d8_wideresnet_seed{args.seed}"
    if args.train:
        train_cifar100_model(
            random_seed=args.seed,
            batch_size=args.batch_size,
            use_wandb=args.use_wandb,
            model_name=model_name,
            max_epochs=args.max_epochs,
        )
    match args.name:
        case "feature_importance":
            feature_importance(
                random_seed=args.seed,
                batch_size=args.batch_size,
                model_name=model_name,
                plot=args.plot,
                n_test=args.n_test,
            )
        case "example_importance":
            example_importance(
                random_seed=args.seed,
                batch_size=args.batch_size,
                model_name=model_name,
                plot=args.plot,
                n_test=args.n_test,
            )
        case other:
            raise NotImplementedError(f"Experiment {args.name} does not exist.")
