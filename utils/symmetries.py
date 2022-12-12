import random

import torch
import torch.nn as nn


class Translation1D(nn.Module):
    def __init__(self, n_steps: int = None):
        super().__init__()
        self.n_steps = n_steps

    def forward(self, x):
        T = x.shape[-1]
        if not self.n_steps:
            n_steps = random.randint(0, T-1)
        else:
            n_steps = self.n_steps
        perm_ids = torch.arange(0, T).to(x.device)
        perm_ids = perm_ids - n_steps
        perm_ids = perm_ids % T
        return x[:, :, perm_ids]