import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import logging
import json
from torch_geometric.nn import GraphConv, global_add_pool
from torch_geometric.loader import DataLoader
from utils.metrics import AverageMeter
from tqdm import tqdm


class ClassifierMutagenicity(torch.nn.Module):
    def __init__(self, dim: int, name: str = 'model') -> None:
        super(ClassifierMutagenicity, self).__init__()
        num_features = 14
        self.latent_dim = dim
        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, 2)
        self.criterion = F.nll_loss
        self.checkpoints_files = []
        self.name = name

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def representation(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        return x

    def representation_to_output(self, x, edge_index, batch, edge_weight=None):
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def last_layer(self) -> nn.Module or None:
        return self.lin2

    def train_epoch(self, device: torch.device, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> np.ndarray:
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
        for data in train_bar:
            data = data.to(device)
            optimizer.zero_grad()
            loss = self.criterion(self.forward(data.x, data.edge_index, data.batch), data.y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), data.num_graphs)
            train_bar.set_description(f"Training Loss {loss_meter.avg:3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader) -> tuple:
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
            for data in dataloader:
                data = data.to(device)
                pred_batch = self.forward(data.x, data.edge_index, data.batch)
                loss = self.criterion(pred_batch, data.y)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(data.y == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / data.num_graphs
                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(
            self,
            device: torch.device,
            train_loader: DataLoader,
            test_loader: DataLoader,
            save_dir: pathlib.Path,
            lr: int = 1e-03,
            n_epoch: int = 200,
            patience: int = 20,
            checkpoint_interval: int = -1,
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
            train_loss = self.train_epoch(device, train_loader, optim)
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