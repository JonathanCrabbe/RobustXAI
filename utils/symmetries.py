import random
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Symmetry(nn.Module, ABC):
    def __int__(self):
        super().__int__()

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def sample_symmetry(self, x):
        ...

    @abstractmethod
    def get_all_symmetries(self, x):
        ...

    @abstractmethod
    def set_symmetry(self, symmetry_param):
        ...


class Translation1D(Symmetry):
    def __init__(self, n_steps: int = None):
        super().__init__()
        self.n_steps = n_steps

    def forward(self, x):
        T = x.shape[-1]
        if not self.n_steps:
            self.sample_symmetry(x)
        perm_ids = torch.arange(0, T).to(x.device)
        perm_ids = perm_ids - self.n_steps
        perm_ids = perm_ids % T
        return x[:, :, perm_ids]

    def sample_symmetry(self, x):
        T = x.shape[-1]
        self.n_steps = random.randint(0, T-1)

    def get_all_symmetries(self, x):
        T = x.shape[-1]
        return list(range(T))

    def set_symmetry(self, n_steps):
       self.n_steps = n_steps


class GraphPermutation(nn.Module):
    def __init__(self, perm: list = None):
        super().__init__()
        self.perm = perm if perm else []

    def forward(self, data):
        num_nodes = data.num_nodes
        if len(self.perm) != num_nodes:
            self.sample_symmetry(data)
        new_data = data.clone().cpu()
        new_data.x = new_data.x[self.perm, :]
        perm_lambda = lambda x: self.perm[x]
        new_data.edge_index.apply_(perm_lambda)
        return new_data.to(data.x.device)

    def sample_symmetry(self, data):
        num_nodes = data.num_nodes
        self.perm = torch.randperm(num_nodes)

    def get_all_symmetries(self, data):
        return None

    def set_symmetry(self, perm: list):
        self.perm = perm