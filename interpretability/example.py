import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from captum.influence import TracInCP, TracInCPFast
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


class ExampleBasedExplainer(nn.Module, ABC):
    def __init__(self, model: nn.Module, X_train: torch.tensor, **kwargs):
        super().__init__()
        self.model = model
        self.X_train = X_train

    @abstractmethod
    def forward(self, x, y):
        ...


class SimplEx(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor):
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


class TracIN(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor,  Y_train: torch.Tensor,
                 loss_function: callable, batch_size: int = 1):
        super().__init__(model, X_train)
        train_subset = TensorDataset(X_train, Y_train)
        last_layer = model.last_layer()
        if last_layer:
            loss_function.reduction = 'sum'
            self.explainer = TracInCPFast(model, last_layer, train_subset, model.checkpoints_files,
                                          loss_fn=loss_function, batch_size=batch_size)
        else:
            self.explainer = TracInCP(model, train_subset, model.checkpoints_files, loss_fn=loss_function,
                                      batch_size=batch_size)

    def forward(self, x, y):
        return self.explainer.influence(x, y)


class InfluenceFunctions(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor,  Y_train: torch.Tensor, train_loader: DataLoader,
                 loss_function: callable, batch_size: int, save_dir: Path):
        super().__init__(model, X_train)
        self.last_layer = model.last_layer()
        self.train_loader = train_loader
        self.Y_train = Y_train
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.ihvp = False
        train_subset = TensorDataset(X_train, Y_train)
        self.subtrain_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
        self.device = self.X_train.device
        self.save_dir = save_dir
        if not save_dir.exists():
            os.makedirs(save_dir)

    def evaluate_ihvp(self, recursion_depth: int = 100,  damp: float = 1e-3, scale: float = 1000,) -> None:
        for train_idx, (x_train, y_train) in enumerate(self.subtrain_loader):
            x_train = x_train.to(self.device)
            loss = self.loss_function(self.model(x_train), y_train)
            grad = self.direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True))
            ihvp = grad.detach().clone()
            train_sampler = iter(self.train_loader)
            for _ in range(recursion_depth):
                X_sample, Y_sample = next(train_sampler)
                X_sample, Y_sample = X_sample.to(self.device), Y_sample.to(self.device)
                sampled_loss = self.loss_function(self.model(X_sample), Y_sample)
                ihvp_prev = ihvp.detach().clone()
                hvp = self.direct_sum(self.hessian_vector_product(sampled_loss, ihvp_prev))
                ihvp = grad + (1 - damp) * ihvp - hvp / scale
            ihvp = ihvp / (scale * len(self.train_loader.dataset))  # Rescale Hessian-Vector products
            torch.save(ihvp.detach().cpu(), self.save_dir / f"train_ihvp{train_idx}.pt")
        self.ihvp = True

    def forward(self, x, y):
        if not self.ihvp:
            self.evaluate_ihvp()
        attribution = torch.zeros((len(x), len(self.X_train)))
        test_subset = TensorDataset(x, y)
        subtest_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
        for test_idx, (x_test, y_test) in enumerate(subtest_loader):
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            test_loss = self.loss_function(self.model(x_test), y_test)
            test_grad = self.direct_sum(torch.autograd.grad(test_loss, self.last_layer.parameters(), create_graph=True))
            test_grad = test_grad.detach().cpu()
            for train_idx in range(len(self.X_train)):
                ihvp = torch.load(self.save_dir / f"train_ihvp{train_idx}.pt")
                attribution[test_idx, train_idx] = torch.dot(ihvp, test_grad)
        return attribution

    def hessian_vector_product(self, loss: torch.Tensor, v: torch.Tensor):
        """
        Multiplies the Hessians of the loss of a model with respect to its parameters by a vector v.
        Adapted from: https://github.com/kohpangwei/influence-release
        This function uses a backproplike approach to compute the product between the Hessian
        and another vector efficiently, which even works for large Hessians with O(p) compelxity for p parameters.
        Arguments:
            loss: scalar/tensor, for example the output of the loss function
            model: the model for which the Hessian of the loss is evaluated
            v: list of torch tensors, rnn.parameters(),
                will be multiplied with the Hessian
        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.
        """

        # First backprop
        first_grads = self.direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(),retain_graph=True, create_graph=True))

        # Elementwise products
        elemwise_products = torch.dot(first_grads.flatten(), v.flatten())

        # Second backprop
        HVP_ = torch.autograd.grad(elemwise_products, self.last_layer.parameters())
        self.model.zero_grad()
        return HVP_

    @staticmethod
    def direct_sum(input_tensors):
        """
        Takes a list of tensors and stacks them into one tensor
        """
        unrolled = [tensor.flatten() for tensor in input_tensors]
        return torch.cat(unrolled)



