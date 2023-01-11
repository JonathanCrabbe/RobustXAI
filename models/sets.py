import numpy as np
import torch
import torch.nn as nn
import pathlib
import logging
import json
from tqdm import tqdm
from utils.metrics import AverageMeter


"""
The models in this file are adapted from https://github.com/manzilzaheer/DeepSets
"""


class PermEqui1_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x


class ClassifierModelNet40(nn.Module):

  def __init__(self, latent_dim, x_dim=3, name: str = 'model'):
    super(ClassifierModelNet40, self).__init__()
    self.name = name
    self.latent_dim = latent_dim
    self.x_dim = x_dim
    self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.latent_dim),
          nn.Tanh(),
          PermEqui1_max(self.latent_dim, self.latent_dim),
          nn.Tanh(),
          PermEqui1_max(self.latent_dim, self.latent_dim),
          nn.Tanh(),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.latent_dim, self.latent_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.latent_dim, 40),
    )
    self.criterion = nn.CrossEntropyLoss()
    self.checkpoints_files = []

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output

  def train_epoch(
          self,
          device: torch.device,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.MultiStepLR,
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
      for x, y in train_bar:
          x, y = x.to(device), y.to(device)
          y_pred = self.forward(x)
          loss = self.criterion(y_pred, y)
          optimizer.zero_grad()
          loss.backward()
          clip_grad(self, 5)
          optimizer.step()
          loss_meter.update(loss.item(), len(x))
          train_bar.set_description(f"Training Loss {loss_meter.avg:3g}")
          train_loss.append(loss.detach().cpu().numpy())
      scheduler.step()
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
          for x, y in dataloader:
              x, y = x.to(device), y.to(device)
              y_pred = self.forward(x)
              loss = self.criterion(y_pred, y)
              test_loss.append(loss.cpu().numpy())
              test_acc.append(
                  torch.count_nonzero(y == torch.argmax(y_pred, dim=-1))
                  .cpu()
                  .numpy()
                  / len(y_pred)
              )

      return np.mean(test_loss), np.mean(test_acc)

  def fit(
          self,
          device: torch.device,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          save_dir: pathlib.Path,
          lr: int = 1e-03,
          n_epoch: int = 1000,
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
      optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-07, eps=1e-3)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(range(400, n_epoch, 400)), gamma=0.1)
      waiting_epoch = 0
      best_test_acc = 0.0
      for epoch in range(n_epoch):
          train_loss = self.train_epoch(device, train_loader, optim, scheduler)
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

  def last_layer(self) -> nn.Module or None:
      return self.ro[-1]


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm