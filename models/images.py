import torch
import torch.nn as nn
import logging
import pathlib
import json
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.symmetries import Translation2D
from tqdm import tqdm
from utils.metrics import AverageMeter
from random import randint
from abc import ABC, abstractmethod
from typing import Tuple
from utils.gcnn import (
    conv1x1,
    conv3x3,
    conv5x5,
    FIBERS,
    init,
)
from e2cnn import nn as e2nn
from e2cnn import gspaces


class ClassifierFashionMnist(ABC, nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierFashionMnist, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.checkpoints_files = []
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def representation(self, x):
        ...

    @abstractmethod
    def last_layer(self) -> nn.Module or None:
        ...

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        augmentation: bool,
    ) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer
        Returns:
            average loss on the training set
        """
        self.train()
        train_loss = []
        loss_meter = AverageMeter("Loss")
        train_bar = tqdm(dataloader, unit="batch", leave=False)
        for images_batch, label_batch in train_bar:
            images_batch = images_batch.to(device)
            label_batch = label_batch.to(device)
            H, W = images_batch.shape[-2:]
            if augmentation:
                transl = Translation2D(randint(0, H), randint(0, W))
                images_batch = transl(images_batch)
            pred_batch = self.forward(images_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(images_batch))
            train_bar.set_description(f"Training Loss {loss_meter.avg:3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(
        self, device: torch.device, dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        One epoch of the testing loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader
        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for images_batch, label_batch in dataloader:
                images_batch = images_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(images_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / len(label_batch)
                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        lr: int = 1e-03,
        n_epoch: int = 200,
        patience: int = 20,
        checkpoint_interval: int = -1,
        augmentation: bool = True,
    ) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save
            augmentation: True if one wants to augment the data with translations
        """
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_acc = 0.0
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim, augmentation)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
                f"Test Loss {test_loss:.3g} \t"
                f"Test Accuracy {test_acc * 100:.3g}% \t "
            )
            if test_acc <= best_test_acc:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_acc = test_acc
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                self.checkpoints_files.append(str(path_to_checkpoint))
                torch.save(self.state_dict(), path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f"Early stopping activated")
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        self.save_metadata(directory)
        path_to_model = directory / (self.name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """
        Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        self.latent_dim = metadata["latent_dim"]
        self.checkpoints_files = metadata["checkpoint_files"]
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """
        Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {
            "latent_dim": self.latent_dim,
            "name": self.name,
            "checkpoint_files": self.checkpoints_files,
        }
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


class AllCNN(ClassifierFashionMnist):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(AllCNN, self).__init__(latent_dim, name)
        self.cnn1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.cnn2 = nn.Conv2d(
            16, 64, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.cnn3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.fc1 = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, 10)
        self.leaky_relu1 = nn.LeakyReLU(inplace=False)
        self.leaky_relu2 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = torch.mean(x, dim=(-2, -1))
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.out(x)
        return x

    def representation(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = torch.mean(x, dim=-1)
        x = self.fc1(x)
        return x

    def representation_to_output(self, x):
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.out(x)
        return x

    def last_layer(self) -> nn.Module or None:
        return self.out


class StandardCNN(ClassifierFashionMnist):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(StandardCNN, self).__init__(latent_dim, name)
        self.cnn1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.cnn2 = nn.Conv2d(
            16, 64, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.cnn3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )
        self.fc1 = nn.Linear(294912, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, 10)
        self.leaky_relu1 = nn.LeakyReLU(inplace=False)
        self.leaky_relu2 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.out(x)
        return x

    def representation(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = torch.mean(x, dim=-1)
        x = self.fc1(x)
        return x

    def representation_to_output(self, x):
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.out(x)
        return x

    def last_layer(self) -> nn.Module or None:
        return self.out


class WideBasic(e2nn.EquivariantModule):
    def __init__(
        self,
        in_fiber: e2nn.FieldType,
        inner_fiber: e2nn.FieldType,
        dropout_rate,
        stride=1,
        out_fiber: e2nn.FieldType = None,
        F: float = 1.0,
        sigma: float = 0.45,
    ):
        super(WideBasic, self).__init__()

        if out_fiber is None:
            out_fiber = in_fiber

        self.in_type = in_fiber
        inner_class = inner_fiber
        self.out_type = out_fiber

        if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.rotation_order
        elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.order()
        else:
            rotations = 0

        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5

        self.bn1 = e2nn.InnerBatchNorm(self.in_type)
        self.relu1 = e2nn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_class, sigma=sigma, F=F, initialize=False)

        self.bn2 = e2nn.InnerBatchNorm(inner_class)
        self.relu2 = e2nn.ReLU(inner_class, inplace=True)

        self.dropout = e2nn.PointwiseDropout(inner_class, p=dropout_rate)

        self.conv2 = conv(
            inner_class,
            self.out_type,
            stride=stride,
            sigma=sigma,
            F=F,
            initialize=False,
        )

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(
                self.in_type,
                self.out_type,
                stride=stride,
                bias=False,
                sigma=sigma,
                F=F,
                initialize=False,
            )
            # if rotations in [0, 2, 4]:
            #     self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
            # else:
            #     self.shortcut = conv3x3(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)

    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x

        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Wide_ResNet(pl.LightningModule):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 7,
        dropout_rate: float = 0.3,
        num_classes=100,
        N: int = 8,
        r: int = 1,
        f: bool = True,
        main_fiber: str = "regular",
        inner_fiber: str = "regular",
        F: float = 1.0,
        sigma: float = 0.45,
        deltaorth: bool = False,
        fixparams: bool = True,
        initial_stride: int = 1,
        conv2triv: bool = True,
    ):
        super(Wide_ResNet, self).__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.distributed = False
        self._fixparams = fixparams
        self.conv2triv = conv2triv

        self._layer = 0
        self._N = N

        # if the model is [F]lip equivariant
        self._f = f

        # level of [R]estriction:
        #   r < 0 : never do restriction, i.e. initial group (either D8 or C8) preserved for the whole network
        #   r = 0 : do restriction before first layer, i.e. initial group doesn't have rotation equivariance (C1 or D1)
        #   r > 0 : restrict after every block, i.e. start with 8 rotations, then restrict to 4 and finally 1
        self._r = r

        self._F = F
        self._sigma = sigma

        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)

        if self._r == 0:
            id = (0, 1) if self._f else 1
            self.gspace, _, _ = self.gspace.restrict(id)

        r1 = e2nn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.in_type = r1

        # r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=self._fixparams)
        r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=True)
        self._in_type = r2

        self.conv1 = conv5x5(r1, r2, sigma=sigma, F=F, initialize=False)
        self.layer1 = self._wide_layer(
            WideBasic,
            nStages[1],
            n,
            dropout_rate,
            stride=initial_stride,
            main_fiber=main_fiber,
            inner_fiber=inner_fiber,
        )
        if self._r > 0:
            id = (0, 4) if self._f else 4
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x

        self.layer2 = self._wide_layer(
            WideBasic,
            nStages[2],
            n,
            dropout_rate,
            stride=2,
            main_fiber=main_fiber,
            inner_fiber=inner_fiber,
        )
        if self._r > 1:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x

        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = None

        self.layer3 = self._wide_layer(
            WideBasic,
            nStages[3],
            n,
            dropout_rate,
            stride=2,
            main_fiber=main_fiber,
            inner_fiber=inner_fiber,
            out_fiber=out_fiber,
        )

        self.bn1 = e2nn.InnerBatchNorm(self.layer3.out_type, momentum=0.9)
        if self.conv2triv:
            self.relu = e2nn.ReLU(self.bn1.out_type, inplace=True)
        else:
            self.mp = e2nn.GroupPooling(self.layer3.out_type)
            self.relu = e2nn.ReLU(self.mp.out_type, inplace=True)

        self.linear = torch.nn.Linear(self.relu.out_type.size, num_classes)

        for name, module in self.named_modules():
            if isinstance(module, e2nn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(
                        module.weights.data, module.basisexpansion
                    )
                else:
                    init.generalized_he_init(module.weights.data, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(e2nn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(e2nn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace

        restrict_layer = e2nn.SequentialModule(*layers)
        return restrict_layer

    def _wide_layer(
        self,
        block,
        planes: int,
        num_blocks: int,
        dropout_rate: float,
        stride: int,
        main_fiber: str = "regular",
        inner_fiber: str = "regular",
        out_fiber: str = None,
    ):
        self._layer += 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](
            self.gspace, planes, fixparams=self._fixparams
        )
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)

        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(
                    self._in_type,
                    inner_class,
                    dropout_rate,
                    stride,
                    out_fiber=out_f,
                    sigma=self._sigma,
                    F=self._F,
                )
            )
            self._in_type = out_f
        return e2nn.SequentialModule(*layers)

    def features(self, x):
        x = e2nn.GeometricTensor(x, self.in_type)

        out = self.conv1(x)

        x1 = self.layer1(out)

        if self.distributed:
            x1.tensor = x1.tensor.cuda(1)

        x2 = self.layer2(self.restrict1(x1))

        if self.distributed:
            x2.tensor = x2.tensor.cuda(2)

        x3 = self.layer3(self.restrict2(x2))
        # out = self.relu(self.mp(self.bn1(out)))

        return x1, x2, x3

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.in_type)

        out = self.conv1(x)
        out = self.layer1(out)

        if self.distributed:
            out.tensor = out.tensor.cuda(1)

        out = self.layer2(self.restrict1(out))

        if self.distributed:
            out.tensor = out.tensor.cuda(2)

        out = self.layer3(self.restrict2(out))

        if self.distributed:
            out.tensor = out.tensor.cuda(3)

        out = self.bn1(out)
        if not self.conv2triv:
            out = self.mp(out)
        out = self.relu(out)

        out = out.tensor

        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def distribute(self):
        self.distributed = True

        self.conv1 = self.conv1.cuda(0)
        self.layer1 = self.layer1.cuda(0)

        if self._r:
            self.restrict1 = self.restrict1.cuda(1)
        self.layer2 = self.layer2.cuda(1)

        if self._r:
            self.restrict2 = self.restrict2.cuda(2)
        self.layer3 = self.layer3.cuda(2)

        self.relu = self.relu.cuda(3)
        self.bn1 = self.bn1.cuda(3)
        # self.mp = self.mp.cuda(3)
        self.avgpool = self.avgpool.cuda(3)
        self.linear = self.linear.cuda(3)

        return self

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.cross_entropy(yhat, y)
        acc = torch.sum(y == torch.argmax(yhat, dim=-1)) / len(y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_lr",
            self.lr_schedulers().get_last_lr()[0],
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.cross_entropy(yhat, y)
        acc = torch.sum(y == torch.argmax(yhat, dim=-1)) / len(y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        loss = F.cross_entropy(yhat, y)
        acc = torch.sum(y == torch.argmax(yhat, dim=-1)) / len(y)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 60,
            "strict": True,
            "name": "exp_lr_scheduler",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
