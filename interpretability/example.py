import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ExampleBasedExplainer(nn.Module, ABC):
    def __init__(self, model: nn.Module, X_train: torch.tensor):
        super().__init__()
        self.model = model
        self.X_train = X_train

    @abstractmethod
    def forward(self, x, y):
        ...


class SimplEx(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor = None):
        super().__init__(model, X_train)
        self.H_train = self.model.representation(self.X_train).detach().flatten(start_dim=1)

    def forward(self, x, y) -> torch.Tensor:
        h = self.model.representation(x).detach().flatten(start_dim=1)
        attribution = self.compute_weights(h, self.H_train)
        return attribution

    @staticmethod
    def compute_weights(
            h: torch.Tensor,
            H_train: torch.Tensor,
            n_epoch: int = 1000,
    ) -> torch.Tensor:
        preweights = torch.zeros((len(h), len(H_train)), requires_grad=True, device=H_train.device)
        optimizer = torch.optim.Adam([preweights])
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            H_approx = torch.einsum("ij,jk->ik", weights, H_train)
            error = ((H_approx - h) ** 2).sum()
            error.backward()
            optimizer.step()
        return torch.softmax(preweights, dim=-1).detach()


class RepresentationSimilarity(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor):
        super().__init__(model, X_train)
        self.H_train = self.model.representation(self.X_train).detach().flatten(start_dim=1).unsqueeze(0)

    def forward(self, x, y) -> torch.Tensor:
        h_test = self.model.representation(x).detach().flatten(start_dim=1).unsqueeze(1)
        attribution = F.cosine_similarity(self.H_train, h_test, dim=-1)
        return attribution