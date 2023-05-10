import random
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from e2cnn import gspaces
from e2cnn import nn as e2nn
from torchvision.transforms.functional import rotate
from typing import Tuple


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
        if self.n_steps is None:
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
    def __init__(self, max_displacement: int, h: int = None, w: int = None):
        super().__init__()
        self.max_displacement = max_displacement
        self.h = h
        self.w = w

    def forward(self, x):
        W, H = x.shape[-2:]
        if self.h is None or self.w is None:
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
        return [(0, 0)] + [
            (w, h)
            for w in range(-self.max_displacement, self.max_displacement + 1)
            for h in range(-self.max_displacement, self.max_displacement + 1)
            if (w, h) != (0, 0)
        ]

    def set_symmetry(self, displacement):
        w, h = displacement
        self.w = w
        self.h = h


class Dihedral(Symmetry):
    def __init__(self, order: int = 4, n_chanels: int = 3):
        super().__init__()
        self.gspace = gspaces.FlipRot2dOnR2(N=order)
        self.n_chanels = n_chanels
        self.in_type = self.in_type = e2nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.n_chanels
        )
        self.group_element = None

    def forward(self, x):
        if self.group_element is None:
            self.sample_symmetry(x)
        x = x.float()
        x = e2nn.GeometricTensor(x, self.in_type)
        x = x.transform(self.group_element)
        return x.tensor

    def sample_symmetry(self, x):
        group_element = random.sample(list(self.gspace.testing_elements), 1)[0]
        self.set_symmetry(group_element)

    def get_all_symmetries(self, x):
        return list(self.in_type.testing_elements)

    def set_symmetry(self, symmetry_param):
        self.group_element = symmetry_param


class AnchoredTranslation2D(Translation2D):
    """
    This class implements a 2D translation symmetry with an anchor point.
    Let g1 be the translation corresponding to the anchor point.
    Let g2 be the translation corresponding to the displacement (w, h).
    This class applies the symmetry g1 o g2 o g1^{-1}.
    This avoids the resulting translation to go out of scope by wrapping the product (g1 o g2).
    Note that the forward assumes that g1 was already applied to the input image x.
    """

    def __init__(self, max_dispacement: int, h: int = 0, w: int = 0):
        super().__init__(max_dispacement, h, w)
        self.set_anchor_point((0, 0))

    def set_anchor_point(self, anchor_point: Tuple[int, int]) -> None:
        self.anchor_point = anchor_point

    def set_symmetry(self, displacement):
        w, h = displacement
        x, y = self.anchor_point
        self.w = self.wrap_coord(x + w) - x
        self.h = self.wrap_coord(y + h) - y

    def wrap_coord(self, x):
        res = x
        if res > self.max_displacement:
            res -= 2 * self.max_displacement + 1
        elif res < -self.max_displacement:
            res += 2 * self.max_displacement + 1
        assert -self.max_displacement <= res <= self.max_displacement
        return res
