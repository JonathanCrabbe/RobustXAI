import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from itertools import product


def l1_distance(x1: torch.Tensor, x2: torch.Tensor):
    ...


def invariance(function: nn.Module, symmetry: nn.Module, data_loader: DataLoader, device: torch.device,
               distance: nn.Module = nn.L1Loss(reduction='none',), N_samp: int = 50) -> torch.Tensor:
    invariance_scores = torch.zeros((len(data_loader.dataset), N_samp))
    for n_samp in tqdm(range(N_samp), leave=False):
        new_scores = []
        for x, _ in data_loader:
            x = x.to(device)
            new_scores.append(distance(function(x), function(symmetry(x))).cpu())
        new_scores = torch.cat(new_scores)
        print(new_scores.shape)
        invariance_scores[:, n_samp] = new_scores
    return torch.mean(invariance_scores, dim=-1)

