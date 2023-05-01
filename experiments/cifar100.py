import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from models.images import Wide_ResNet
from datasets.loaders import Cifar100Dataset
from pathlib import Path


data_dir = Path.cwd() / "datasets/cifar100"
save_dir = Path.cwd() / "results/cifar100"
model = Wide_ResNet()
datamodule = Cifar100Dataset(data_dir=data_dir)
wdb_logger = pl.loggers.WandbLogger(
    project="RobustXAI", name="cifar100_e2wrn_test", save_dir=save_dir
)
trainer = pl.Trainer(logger=wdb_logger, max_epochs=200, default_root_dir=save_dir)
tuner = Tuner(trainer)
# tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
trainer.fit(model, datamodule=datamodule)
