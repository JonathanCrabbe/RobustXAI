import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models.images import Wide_ResNet
from datasets.loaders import Cifar100Dataset
from pathlib import Path
from utils.misc import set_random_seed


def train_cifar100_model(
    random_seed: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/cifar100/",
    data_dir: Path = Path.cwd() / "datasets/cifar100",
    use_wandb: bool = False,
) -> None:
    set_random_seed(random_seed)
    model_dir = model_dir / model_name
    model = Wide_ResNet()
    datamodule = Cifar100Dataset(data_dir=data_dir, batch_size=batch_size)
    logger = (
        pl.loggers.WandbLogger(project="RobustXAI", name=model_name, save_dir=model_dir)
        if use_wandb
        else None
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            monitor="val/acc",
            every_n_epochs=10,
            save_top_k=-1,
            filename=model_name + "-{epoch:02d}-{val_acc:.2f}",
        ),
        EarlyStopping(monitor="val/acc", patience=10, mode="max"),
    ]
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=200,
        default_root_dir=model_dir,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    print(trainer.test(model, ckpt_path="best", datamodule=datamodule))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    model_name = f"cifar100_e2wrn_seed{args.seed}"
    if args.train:
        train_cifar100_model(
            random_seed=args.seed,
            batch_size=args.batch_size,
            use_wandb=args.use_wandb,
            model_name=model_name,
        )
