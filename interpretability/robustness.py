import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.symmetries import Symmetry, AnchoredTranslation2D, Translation2D
from random import shuffle
from captum.metrics import sensitivity_max
from captum.attr import Attribution, GradientShap, Occlusion


def l1_distance(
    x1: torch.Tensor, x2: torch.Tensor, reduce: bool = False
) -> torch.Tensor:
    d = torch.sum(torch.flatten(torch.abs(x1 - x2), start_dim=1), dim=-1)
    if reduce:
        d = torch.mean(d)
    return d


def cos_similarity(
    x1: torch.Tensor, x2: torch.Tensor, reduce: bool = False
) -> torch.Tensor:
    s = F.cosine_similarity(
        torch.flatten(x1, start_dim=1) + 1e-4, torch.flatten(x2, start_dim=1) + 1e-4
    )  # Small offset for numerical stability
    if reduce:
        s = torch.mean(s)
    return s


def accuracy(x1: torch.Tensor, x2: torch.Tensor, reduce: bool = False) -> torch.Tensor:
    s = torch.mean(torch.where(x1 == x2, 1.0, 0.0), dim=-1)
    if reduce:
        s = torch.mean(s)
    return s


def model_invariance(
    model: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for x, _ in data_loader:
            x = x.to(device)
            y1 = model(x)
            symmetry.sample_symmetry(x)
            y2 = model(symmetry(x))
            sample_scores.append(similarity(y1, y2).detach().cpu())
        sample_scores = torch.cat(sample_scores)
        invariance_scores[:, sample_id] = sample_scores
    if reduce:
        invariance_scores = torch.mean(invariance_scores, dim=-1)
    return invariance_scores


def model_invariance_exact(
    model: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
) -> torch.Tensor:
    invariance_scores = []
    for x, _ in tqdm(data_loader, leave=False, unit="batch"):
        batch_scores = None
        x = x.to(device)
        y1 = model(x)
        for param in tqdm(symmetry.get_all_symmetries(x), leave=False, unit="symmetry"):
            symmetry.set_symmetry(param)
            y2 = model(symmetry(x))
            if batch_scores is None:
                batch_scores = similarity(y1, y2).detach().cpu()
            else:
                batch_scores += similarity(y1, y2).detach().cpu()
        invariance_scores.append(batch_scores / len(symmetry.get_all_symmetries(x)))
    invariance_scores = torch.cat(invariance_scores)
    return invariance_scores


def graph_model_invariance(
    model: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for data in tqdm(data_loader, leave=False, unit="graph"):
            data = data.to(device)
            symmetry.sample_symmetry(data)
            new_data = symmetry(data)
            y1 = model(data.x, data.edge_index, data.batch)
            y2 = model(new_data.x, new_data.edge_index, new_data.batch)
            sample_scores.append(similarity(y1, y2).detach().cpu())
        sample_scores = torch.cat(sample_scores)
        invariance_scores[:, sample_id] = sample_scores
    if reduce:
        invariance_scores = torch.mean(invariance_scores, dim=-1)
    return invariance_scores


def explanation_invariance(
    explainer: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for x, y in tqdm(data_loader, leave=False, unit="batch"):
            x = x.to(device)
            y = y.to(device)
            e1 = explainer(x, y)
            symmetry.sample_symmetry(x)
            e2 = explainer(symmetry(x), y)
            sample_scores.append(similarity(e1, e2).detach().cpu())
        sample_scores = torch.cat(sample_scores)
        invariance_scores[:, sample_id] = sample_scores
    if reduce:
        invariance_scores = torch.mean(invariance_scores, dim=-1)
    return invariance_scores


def explanation_invariance_exact(
    explainer: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
) -> torch.Tensor:
    invariance_scores = []
    for x, y in tqdm(data_loader, leave=False, unit="batch"):
        batch_scores = None
        x, y = x.to(device), y.to(device)
        if isinstance(symmetry, Translation2D) and isinstance(
            explainer, InvariantExplainer
        ):
            explainer.anchor = (0, 0)
        e1 = explainer(x, y)
        for param in tqdm(symmetry.get_all_symmetries(x), leave=False, unit="symmetry"):
            symmetry.set_symmetry(param)
            if isinstance(symmetry, Translation2D) and isinstance(
                explainer, InvariantExplainer
            ):
                explainer.anchor = param
            e2 = explainer(symmetry(x), y)
            sim = similarity(e1, e2).detach().cpu()
            if batch_scores is None:
                batch_scores = sim
            else:
                batch_scores += sim
        invariance_scores.append(batch_scores / len(symmetry.get_all_symmetries(x)))
    invariance_scores = torch.cat(invariance_scores)
    return invariance_scores


def graph_explanation_invariance(
    explanation: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for data in tqdm(data_loader, leave=False, unit="graph"):
            data = data.to(device)
            symmetry.sample_symmetry(data)
            new_data = symmetry(data)
            e1 = explanation(data)
            e2 = explanation(new_data)
            sample_scores.append(similarity(e1, e2).detach().cpu())
        sample_scores = torch.cat(sample_scores)
        invariance_scores[:, sample_id] = sample_scores
    if reduce:
        invariance_scores = torch.mean(invariance_scores, dim=-1)
    return invariance_scores


def explanation_equivariance(
    explainer: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    equivariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for x, y in tqdm(data_loader, leave=False, unit="batch"):
            x = x.to(device)
            y = y.to(device)
            symmetry.sample_symmetry(x)
            e1 = symmetry(explainer(x, y))
            e2 = explainer(symmetry(x), y)
            sample_scores.append(similarity(e1, e2).detach().cpu())
        sample_scores = torch.cat(sample_scores)
        equivariance_scores[:, sample_id] = sample_scores
    if reduce:
        equivariance_scores = torch.mean(equivariance_scores, dim=-1)
    return equivariance_scores


def explanation_equivariance_exact(
    explainer: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
) -> torch.Tensor:
    invariance_scores = []
    for x, y in tqdm(data_loader, leave=False, unit="batch"):
        batch_scores = None
        x, y = x.to(device), y.to(device)
        for param in tqdm(symmetry.get_all_symmetries(x), leave=False, unit="symmetry"):
            symmetry.set_symmetry(param)
            e1 = symmetry(explainer(x, y))
            e2 = explainer(symmetry(x), y)
            if batch_scores is None:
                batch_scores = similarity(e1, e2).detach().cpu()
            else:
                batch_scores += similarity(e1, e2).detach().cpu()
        invariance_scores.append(batch_scores / len(symmetry.get_all_symmetries(x)))
    invariance_scores = torch.cat(invariance_scores)
    return invariance_scores


def graph_explanation_equivariance(
    explainer: nn.Module,
    symmetry: Symmetry,
    data_loader: DataLoader,
    device: torch.device,
    similarity: callable = cos_similarity,
    N_samp: int = 50,
    reduce: bool = True,
) -> torch.Tensor:
    invariance_scores = torch.zeros(len(data_loader.dataset), N_samp)
    for sample_id in tqdm(range(N_samp), leave=False, unit="MC sample"):
        sample_scores = []
        for data in tqdm(data_loader, leave=False, unit="graph"):
            data = data.to(device)
            symmetry.sample_symmetry(data)
            new_data = symmetry(data)
            e1 = symmetry.forward_nodes(explainer.forward_graph(data))
            e2 = explainer.forward_graph(new_data)
            sample_scores.append(
                similarity(e1.unsqueeze(0), e2.unsqueeze(0)).detach().cpu()
            )
        sample_scores = torch.cat(sample_scores)
        invariance_scores[:, sample_id] = sample_scores
    if reduce:
        invariance_scores = torch.mean(invariance_scores, dim=-1)
    return invariance_scores


def sensitivity(
    explainer: Attribution, data_loader: DataLoader, device: torch.device
) -> torch.Tensor:
    sens_scores = []
    for x, y in tqdm(data_loader, leave=False, unit="batch"):
        x, y = x.to(device), y.to(device)
        if isinstance(explainer, GradientShap):
            baselines = torch.zeros(x.shape, device=device)
            sens = sensitivity_max(
                explainer.attribute, x, target=y, baselines=baselines
            )
        elif isinstance(explainer, Occlusion):
            window_shapes = (1,) + (len(x.shape) - 2) * (5,)
            sens = sensitivity_max(
                explainer.attribute, x, target=y, sliding_window_shapes=window_shapes
            )
        else:
            sens = sensitivity_max(explainer.attribute, x, target=y)
        sens_scores.append(sens)
    sens_scores = torch.cat(sens_scores, dim=0)
    return sens_scores


class InvariantExplainer(nn.Module):
    def __init__(
        self, explainer: nn.Module, symmetry: Symmetry, N_inv: int, round: bool
    ):
        super().__init__()
        self.explainer = explainer
        self.symmetry = symmetry
        self.N_inv = N_inv
        self.round = round
        self.anchor = (0, 0)

    def forward(self, x, y) -> torch.Tensor:
        if isinstance(self.symmetry, AnchoredTranslation2D):
            self.symmetry.set_anchor_point(self.anchor)
        explanation = self.explainer(x, y)
        if self.symmetry.get_all_symmetries(x):
            params = self.symmetry.get_all_symmetries(x)[1:]
            shuffle(params)
            for param in params[: self.N_inv - 1]:
                self.symmetry.set_symmetry(param)
                explanation += self.explainer(self.symmetry(x), y)
        else:
            for _ in range(self.N_inv - 1):
                self.symmetry.sample_symmetry(x)
                explanation += self.explainer(self.symmetry(x), y)
        explanation = explanation.float() / self.N_inv
        if self.round:
            explanation = torch.round(explanation)
        return explanation
