import logging
import os
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datasets.loaders import IMDBDataset
from models.nlp import BOWClassifier
from utils.misc import set_random_seed


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


if __name__ == "__main__":
    main()
