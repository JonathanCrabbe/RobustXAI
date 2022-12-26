import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from utils.misc import direct_sum


class ExampleBasedExplainer(nn.Module, ABC):
    def __init__(self, model: nn.Module, X_train: torch.tensor, **kwargs):
        super().__init__()
        self.model = model
        self.X_train = X_train

    @abstractmethod
    def forward(self, x, y):
        ...


class SimplEx(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor, **kwargs):
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
    def __init__(self, model: nn.Module,  X_train: torch.Tensor, **kwargs):
        super().__init__(model, X_train)
        self.H_train = self.model.representation(self.X_train).detach().flatten(start_dim=1).unsqueeze(0)

    def forward(self, x, y) -> torch.Tensor:
        H = self.model.representation(x).detach().flatten(start_dim=1).unsqueeze(1)
        attribution = F.cosine_similarity(self.H_train, H, dim=-1)
        return attribution


class TracIN(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor,  Y_train: torch.Tensor,
                 loss_function: callable, save_dir: Path, **kwargs):
        super().__init__(model, X_train)
        self.last_layer = model.last_layer()
        self.save_dir = save_dir/'tracin'
        self.loss_function = loss_function
        self.checkpoints = model.checkpoints_files
        self.device = X_train.device
        train_subset = TensorDataset(X_train, Y_train)
        self.subtrain_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
        self.train_grads = False
        if not self.save_dir.exists():
            os.makedirs(self.save_dir)

    def forward(self, x, y):
        if not self.train_grads:
            self.compute_train_grads()
        attribution = torch.zeros((len(x), len(self.X_train)))
        test_subset = TensorDataset(x, y)
        subtest_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
        for test_idx, (x_test, y_test) in enumerate(subtest_loader):
            test_grad = None
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            for checkpoint in self.checkpoints:
                self.model.load_state_dict(torch.load(checkpoint), strict=False)
                test_loss = self.loss_function(self.model(x_test), y_test)
                if test_grad is not None:
                    test_grad += direct_sum(torch.autograd.grad(test_loss, self.last_layer.parameters(), create_graph=True))
                else:
                    test_grad = direct_sum(torch.autograd.grad(test_loss, self.last_layer.parameters(), create_graph=True))
            test_grad = test_grad.detach().cpu()
            for train_idx in range(len(self.X_train)):
                train_grad = torch.load(self.save_dir/f"train_grad{train_idx}.pt")
                attribution[test_idx, train_idx] = torch.dot(train_grad, test_grad)
        return attribution

    def compute_train_grads(self) -> None:
        for train_idx, (x_train, y_train) in enumerate(self.subtrain_loader):
            grad = None
            for checkpoint in self.checkpoints:
                self.model.load_state_dict(torch.load(checkpoint), strict=False)
                loss = self.loss_function(self.model(x_train), y_train)
                if grad is not None:
                    grad += direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True))
                else:
                    grad = direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True))
            torch.save(grad.detach().cpu(), self.save_dir/f"train_grad{train_idx}.pt")
        self.train_grads = True


class InfluenceFunctions(ExampleBasedExplainer):
    def __init__(self, model: nn.Module,  X_train: torch.Tensor,  Y_train: torch.Tensor, train_loader: DataLoader,
                 loss_function: callable, save_dir: Path, recursion_depth: int,  **kwargs):
        super().__init__(model, X_train)
        self.last_layer = model.last_layer()
        self.train_loader = train_loader
        self.Y_train = Y_train
        self.loss_function = loss_function
        self.ihvp = False
        train_subset = TensorDataset(X_train, Y_train)
        self.subtrain_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
        self.device = self.X_train.device
        self.save_dir = save_dir/'influence_functions'
        self.recursion_depth = recursion_depth
        if not self.save_dir.exists():
            os.makedirs(self.save_dir)

    def evaluate_ihvp(self,  damp: float = 1e-3, scale: float = 1000,) -> None:
        for train_idx, (x_train, y_train) in enumerate(self.subtrain_loader):
            x_train = x_train.to(self.device)
            loss = self.loss_function(self.model(x_train), y_train)
            grad = direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True))
            ihvp = grad.detach().clone()
            train_sampler = iter(self.train_loader)
            for _ in range(self.recursion_depth):
                X_sample, Y_sample = next(train_sampler)
                X_sample, Y_sample = X_sample.to(self.device), Y_sample.to(self.device)
                sampled_loss = self.loss_function(self.model(X_sample), Y_sample)
                ihvp_prev = ihvp.detach().clone()
                hvp = direct_sum(self.hessian_vector_product(sampled_loss, ihvp_prev))
                ihvp = grad + (1 - damp) * ihvp - hvp / scale
            torch.save(ihvp.detach().cpu(), self.save_dir/f"train_ihvp{train_idx}.pt")
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
            test_grad = direct_sum(torch.autograd.grad(test_loss, self.last_layer.parameters(), create_graph=True))
            test_grad = test_grad.detach().cpu()
            for train_idx in range(len(self.X_train)):
                ihvp = torch.load(self.save_dir/f"train_ihvp{train_idx}.pt")
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
        first_grads = direct_sum(torch.autograd.grad(loss, self.last_layer.parameters(), retain_graph=True, create_graph=True))

        # Elementwise products
        elemwise_products = torch.dot(first_grads.flatten(), v.flatten())

        # Second backprop
        HVP_ = torch.autograd.grad(elemwise_products, self.last_layer.parameters())
        self.model.zero_grad()
        return HVP_



