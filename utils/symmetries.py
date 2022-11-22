import torch
import torch.nn as nn


class Translation1D(nn.Module):
    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = n_steps

    def forward(self, x):
        T = x.shape[-1]
        perm_ids = torch.arange(0, T).to(x.device)
        perm_ids = perm_ids - self.n_steps
        perm_ids = perm_ids % T
        return x[:, :, perm_ids]