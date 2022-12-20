import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from captum.influence import TracInCP, TracInCPFast
from torch.utils.data import TensorDataset, DataLoader


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
        last_layer = model.last_linear_layer()
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
                 loss_function: callable, batch_size: int = 1):
        super().__init__(model, X_train)
        self.last_layer = model.last_linear_layer()
        self.train_loader = train_loader
        self.Y_train = Y_train
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.ihvp = None

    def evaluate_ihvp(self, recursion_depth: int = 100,  damp: float = 1e-3, scale: float = 1000,) -> None:
        losses = [self.loss_function(model(x), y) for x, y in zip(torch.split(self.X_train, 1), torch.split(self.Y_train, 1))]
        grads = [torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True)[0] for loss in losses]
        ihvp = [grad.detach().cpu().clone() for grad in grads]
        train_iter = iter(self.train_loader)
        for _ in range(recursion_depth):
            X_sample, Y_sample = next(train_iter)
            loss_sample = self.loss_function(model(X_sample), Y_sample)
            ihvp_prev = [ihvp[k].detach().clone() for k in range(len(self.X_train))]
            hvps_ = [self.hessian_vector_product(loss_sample, self.model, [ihvp_prev[k]]) for k in range(len(self.X_train))]
            ihvp = [g_ + (1 - damp) * ihvp_ - hvp_ / scale for (g_, ihvp_, hvp_) in zip(grads, ihvp_prev, hvps_)]
        ihvp = [ihvp[k] / (scale * len(self.train_loader.dataset)) for k in range(len(self.X_train))]   # Rescale Hessian-Vector products
        ihvp = torch.stack(ihvp, dim=0) #".reshape((len(train_idx), -1))  # Make a tensor (len(train_idx), n_params)

    def forward(self, x, y):
        if not self.ihvp:
            self.evaluate_ihvp()
        losses = [self.loss_function(model(xi), yi) for xi, yi in zip(torch.split(x, 1), torch.split(y, 1))]
        grads = [torch.autograd.grad(loss, self.last_layer.parameters(), create_graph=True)[0] for loss in losses]
        grads = torch.stack(grads, dim=0)
        attribution = torch.einsum("ab,cb->ac", grads, self.ihvp)
        return attribution

    @staticmethod
    def hessian_vector_product(loss, model, v):
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
        first_grads = stack_torch_tensors(
            torch.autograd.grad(
                loss, model.encoder.parameters(), retain_graph=True, create_graph=True
            )
        )

        # Elementwise products
        elemwise_products = torch.dot(first_grads.flatten(), v.flatten())

        # Second backprop
        HVP_ = torch.autograd.grad(elemwise_products, model.encoder.parameters())

        return HVP_


