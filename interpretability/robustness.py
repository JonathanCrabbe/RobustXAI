import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.symmetries import Symmetry


def l1_distance(x1: torch.Tensor, x2: torch.Tensor, reduce: bool = False) -> torch.Tensor:
    d = torch.sum(torch.flatten(torch.abs(x1 - x2), start_dim=1), dim=-1)
    if reduce:
        d = torch.mean(d)
    return d


def cos_similarity(x1: torch.Tensor, x2: torch.Tensor, reduce: bool = False) -> torch.Tensor:
    return F.cosine_similarity(torch.flatten(x1, start_dim=1), torch.flatten(x2, start_dim=1))


def invariance(function: nn.Module, symmetry: Symmetry, data_loader: DataLoader, device: torch.device,
               distance: callable = l1_distance, N_samp: int = 50) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset))
    batch_size = data_loader.batch_size
    for _ in tqdm(range(N_samp), leave=False, unit='MC sample'):
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            y1 = function(x)
            symmetry.sample_symmetry(x)
            y2 = function(symmetry(x))
            invariance_scores[batch_size*batch_idx:batch_size*batch_idx+len(x)] += distance(y1, y2).detach().cpu()
    return invariance_scores / N_samp


def equivariance(function: nn.Module, symmetry: Symmetry, data_loader: DataLoader, device: torch.device,
                 distance: callable = l1_distance, N_samp: int = 50) -> torch.Tensor:
    equivariance_scores = torch.zeros(len(data_loader.dataset))
    batch_size = data_loader.batch_size
    for _ in tqdm(range(N_samp), leave=False, unit='MC sample'):
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            symmetry.sample_symmetry(x)
            x1 = symmetry(function(x, y))
            x2 = function(symmetry(x), y)
            equivariance_scores[batch_size*batch_idx:batch_size*batch_idx+len(x)] += distance(x1, x2).detach().cpu()
    return equivariance_scores / N_samp

