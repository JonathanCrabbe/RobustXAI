import torch
import torch.nn as nn
import logging
import pathlib
import json
import numpy as np
import torch.nn.functional as F
from utils.symmetries import Translation1D
from tqdm import tqdm
from utils.metrics import AverageMeter
from random import randint
from abc import ABC, abstractmethod


class ClassifierECG(ABC, nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierECG, self).__init__()
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
        augmentation: bool
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
        for series_batch, label_batch in train_bar:
            T = series_batch.shape[-1]
            if augmentation:
                transl = Translation1D(randint(0, T))
                series_batch = transl(series_batch)
            series_batch = series_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(series_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(series_batch))
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
            for series_batch, label_batch in dataloader:
                series_batch = series_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(series_batch)
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
        n_epoch: int = 500,
        patience: int = 50,
        checkpoint_interval: int = -1,
        augmentation: bool = True
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
                path_to_checkpoint = save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
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
        self.latent_dim = metadata['latent_dim']
        self.checkpoints_files = metadata['checkpoint_files']
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


class StandardCNN(ClassifierECG):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(StandardCNN, self).__init__(latent_dim, name)
        self.cnn1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.cnn2 = nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.maxpool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(2944, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 2)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = self.maxpool3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        return x

    def representation(self, x):
        x = self.cnn1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = self.maxpool3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        return x

    def representation_to_output(self, h):
        h = self.out(h)
        return h

    def last_layer(self) -> nn.Module or None:
        return self.out


class AllCNN(ClassifierECG):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(AllCNN, self).__init__(latent_dim, name)
        self.cnn1 = nn.Conv1d(1, latent_dim, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.cnn2 = nn.Conv1d(latent_dim, 2, kernel_size=3, stride=1, padding=1, padding_mode='circular')

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = torch.mean(x, dim=-1)
        return x

    def representation(self, x):
        return F.relu(self.cnn1(x))

    def representation_to_output(self, h):
        h = torch.mean(self.cnn2(h), dim=-1)
        return h

    def last_layer(self) -> nn.Module or None:
        return self.cnn2




