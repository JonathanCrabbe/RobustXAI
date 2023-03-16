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
        self.n_steps = random.randint(0, T - 1)

    def get_all_symmetries(self, x):
        T = x.shape[-1]
        return list(range(T))

    def set_symmetry(self, n_steps):
        self.n_steps = n_steps


class GraphPermutation(Symmetry):
    def __init__(self, perm: list = None):
        super().__init__()
        self.perm = perm if perm else torch.tensor([])

    def forward(self, data):
        num_nodes = data.num_nodes
        if len(self.perm) != num_nodes:
            self.sample_symmetry(data)
        new_data = data.clone().cpu()
        new_data.x = new_data.x[self.perm, :]
        inv_perm = torch.argsort(self.perm)
        perm_lambda = lambda x: inv_perm[x]
        new_data.edge_index.apply_(perm_lambda)
        return new_data.to(data.x.device)

    def forward_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.perm, :]

    def sample_symmetry(self, data):
        num_nodes = data.num_nodes
        self.perm = torch.randperm(num_nodes)

    def get_all_symmetries(self, data):
        raise RuntimeError("Symmetry group too large")

    def set_symmetry(self, perm: list):
        self.perm = perm


class SetPermutation(Symmetry):
    def __init__(self, perm: list = None):
        super().__init__()
        self.perm = perm if perm else torch.tensor([])

    def forward(self, x):
        num_elems = x.shape[1]
        if len(self.perm) != num_elems:
            self.sample_symmetry(x)
        x_new = x[:, self.perm, :]
        return x_new

    def sample_symmetry(self, x):
        num_elems = x.shape[1]
        self.perm = torch.randperm(num_elems)

    def get_all_symmetries(self, data):
        raise RuntimeError("Symmetry group too large")

    def set_symmetry(self, perm: list):
        self.perm = perm


class Translation2D(Symmetry):
    def __init__(self, max_dispacement: int, h: int = None, w: int = None):
        super().__init__()
        self.max_displacement = max_dispacement
        self.h = h
        self.w = w

    def forward(self, x):
        W, H = x.shape[-2:]
        if not self.h or not self.w:
            self.sample_symmetry(x)
        w_perm_ids = torch.arange(0, W).to(x.device)
        h_perm_ids = torch.arange(0, H).to(x.device)
        w_perm_ids = w_perm_ids - self.w
        w_perm_ids = w_perm_ids % W
        h_perm_ids = h_perm_ids - self.h
        h_perm_ids = h_perm_ids % H
        x = x[:, :, :, h_perm_ids]
        x = x[:, :, w_perm_ids, :]
        return x

    def sample_symmetry(self, x):
        self.w = random.randint(-self.max_displacement, self.max_displacement)
        self.h = random.randint(-self.max_displacement, self.max_displacement)

    def get_all_symmetries(self, x):
        return [
            (w, h)
            for w in range(-self.max_displacement, self.max_displacement + 1)
            for h in range(-self.max_displacement, self.max_displacement + 1)
        ]

    def set_symmetry(self, displacement):
        w, h = displacement
        self.w = w
        self.h = h
