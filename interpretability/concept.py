import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from datasets.loaders import ConceptDataset
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.metrics import accuracy_score
from e2cnn.nn import GeometricTensor


class ConceptExplainer(ABC, nn.Module):
    """
    An abstract class that contains the interface for any post-hoc concept explainer
    """

    @abc.abstractmethod
    def __init__(
        self, model: nn.Module, dataset: ConceptDataset, layer: nn.Module, **kwargs
    ):
        super(ConceptExplainer, self).__init__()
        self.model = model
        self.classifiers = None
        self.dataset = dataset
        self.H = None

        def hook(module, input, output):
            # Handle conversion to tensor in case of GeometricTensor
            if isinstance(output, GeometricTensor):
                output = output.tensor
            self.H = output.flatten(start_dim=1).detach().cpu().numpy()

        self.handle = layer.register_forward_hook(hook)

    def remove_hook(self):
        self.handle.remove()

    @abc.abstractmethod
    def fit(self, device: torch.device, concept_set_size: int) -> None:
        """
        Fit a concept classifier for each concept dataset
        """
        ...

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Predicts the presence or absence of concept importance for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        ...


class CAR(ConceptExplainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ConceptDataset,
        layer: nn.Module,
        kernel: str = "rbf",
        **kwargs
    ):
        super(CAR, self).__init__(model, dataset, layer)
        self.kernel = kernel

    def fit(self, device: torch.device, concept_set_size: int = 100) -> None:
        encoders = []
        classifiers = []
        for concept_id, concept_name in enumerate(self.dataset.concept_names()):
            encoder = PCA(10)
            classifier = SVC(kernel=self.kernel)
            X_train, C_train = self.dataset.generate_concept_dataset(
                concept_id, concept_set_size
            )
            self.model(X_train.to(device))
            H_proj = encoder.fit_transform(self.H)
            classifier.fit(H_proj, C_train.numpy())
            encoders.append(encoder)
            classifiers.append(classifier)
        self.encoders = encoders
        self.classifiers = classifiers

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert self.classifiers and self.encoders
        self.model(x)
        C_pred = torch.zeros((len(x), len(self.classifiers)))
        for concept_id, (encoder, classifier) in enumerate(
            zip(self.encoders, self.classifiers)
        ):
            H_proj = encoder.transform(self.H)
            C_pred[:, concept_id] = torch.from_numpy(classifier.predict(H_proj))
        return C_pred

    def concept_accuracy(
        self,
        test_set: ConceptDataset,
        device: torch.device,
        concept_set_size: int = 1000,
    ) -> dict:
        accuracies = {}
        for concept_id, (encoder, classifier) in enumerate(
            zip(self.encoders, self.classifiers)
        ):
            X_test, C_test = test_set.generate_concept_dataset(
                concept_id, concept_set_size
            )
            self.model(X_test.to(device))
            H_test = self.H.copy()
            H_proj = encoder.transform(H_test)
            C_pred = classifier.predict(H_proj)
            accuracies[test_set.concept_names()[concept_id]] = accuracy_score(
                C_test, C_pred
            )
        return accuracies


class CAV(ConceptExplainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ConceptDataset,
        layer: nn.Module,
        n_classes: int,
        **kwargs
    ):
        super(CAV, self).__init__(model, dataset, layer)
        self.n_classes = n_classes

    def fit(self, device: torch.device, concept_set_size: int = 100) -> None:
        classifiers = []
        for concept_id, concept_name in enumerate(self.dataset.concept_names()):
            classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
            X_train, C_train = self.dataset.generate_concept_dataset(
                concept_id, concept_set_size
            )
            self.model(X_train.to(device))
            classifier.fit(self.H, C_train.numpy())
            classifiers.append(classifier)
        self.classifiers = classifiers

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert self.classifiers
        self.model(x)
        C_pred = torch.zeros((len(x), len(self.classifiers)))
        for concept_id, classifier in enumerate(self.classifiers):
            C_pred[:, concept_id] = torch.from_numpy(classifier.predict(self.H))
        return C_pred

    def sensitivity(self, x: torch.tensor, y: torch.Tensor) -> torch.Tensor:
        one_hot_labels = F.one_hot(y, self.n_classes).to(x.device)
        H = self.model.representation(x).requires_grad_()
        Y = self.model.representation_to_output(H)
        grads = torch.autograd.grad(Y, H, grad_outputs=one_hot_labels)[0]
        cavs = self.get_activation_vectors().to(x.device)
        if len(grads.shape) > 2:
            grads = grads.flatten(start_dim=1)
        C_sens = torch.einsum("ci,bi->bc", cavs, grads).detach().cpu()
        return torch.where(C_sens > 0, 1, 0)

    def get_activation_vectors(self):
        assert self.classifiers
        cavs = []
        for classifier in self.classifiers:
            cavs.append(torch.tensor(classifier.coef_).float().reshape(1, -1))
        return torch.cat(cavs, dim=0)

    def concept_accuracy(
        self,
        test_set: ConceptDataset,
        device: torch.device,
        concept_set_size: int = 1000,
    ) -> dict:
        accuracies = {}
        for concept_id, classifier in enumerate(self.classifiers):
            X_test, C_test = test_set.generate_concept_dataset(
                concept_id, concept_set_size
            )
            self.model(X_test.to(device))
            H_test = self.H.copy()
            C_pred = classifier.predict(H_test)
            accuracies[test_set.concept_names()[concept_id]] = accuracy_score(
                C_test, C_pred
            )
        return accuracies


class GraphConceptExplainer(ABC, nn.Module):
    """
    An abstract class that contains the interface for any post-hoc concept explainer
    """

    @abc.abstractmethod
    def __init__(
        self, model: nn.Module, dataset: ConceptDataset, layer: nn.Module, **kwargs
    ):
        super(GraphConceptExplainer, self).__init__()
        self.model = model
        self.classifiers = None
        self.dataset = dataset
        self.H = None

        def hook(module, input, output):
            self.H = output.flatten(start_dim=1).detach().cpu().numpy()

        self.handle = layer.register_forward_hook(hook)

    def remove_hook(self):
        self.handle.remove()

    @abc.abstractmethod
    def fit(self, device: torch.device, concept_set_size: int, batch_size: int) -> None:
        """
        Fit a concept classifier for each concept dataset
        """
        ...

    @abc.abstractmethod
    def forward(self, data: GraphData) -> torch.Tensor:
        """
        Predicts the presence or absence of concept importance for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        ...


class GraphCAR(GraphConceptExplainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ConceptDataset,
        layer: nn.Module,
        kernel: str = "rbf",
        **kwargs
    ):
        super(GraphCAR, self).__init__(model, dataset, layer)
        self.kernel = kernel

    def fit(
        self, device: torch.device, concept_set_size: int = 100, batch_size: int = 200
    ) -> None:
        encoders = []
        classifiers = []
        for concept_id, concept_name in enumerate(self.dataset.concept_names()):
            encoder = PCA(10)
            classifier = SVC(kernel=self.kernel)
            dataset_train, C_train = self.dataset.generate_concept_dataset(
                concept_id, concept_set_size
            )
            train_loader = GraphDataLoader(dataset_train, batch_size, shuffle=False)
            H_train = []
            for data_train in train_loader:
                data_train = data_train.to(device)
                self.model(data_train.x, data_train.edge_index, data_train.batch)
                H_train.append(self.H.copy())
            H_train = np.concatenate(H_train, axis=0)
            H_proj = encoder.fit_transform(H_train)
            classifier.fit(H_proj, C_train.numpy())
            encoders.append(encoder)
            classifiers.append(classifier)
        self.encoders = encoders
        self.classifiers = classifiers

    def forward(self, data: GraphData) -> torch.Tensor:
        assert self.classifiers and self.encoders
        self.model(data.x, data.edge_index, data.batch)
        C_pred = torch.zeros((1, len(self.classifiers)))
        for concept_id, (encoder, classifier) in enumerate(
            zip(self.encoders, self.classifiers)
        ):
            H_proj = encoder.transform(self.H)
            C_pred[:, concept_id] = torch.from_numpy(classifier.predict(H_proj))
        return C_pred

    def concept_accuracy(
        self,
        test_set: ConceptDataset,
        device: torch.device,
        concept_set_size: int = 1000,
        batch_size: int = 200,
    ) -> dict:
        accuracies = {}
        for concept_id, (encoder, classifier) in enumerate(
            zip(self.encoders, self.classifiers)
        ):
            dataset_test, C_test = test_set.generate_concept_dataset(
                concept_id, concept_set_size
            )
            test_loader = GraphDataLoader(dataset_test, batch_size, shuffle=False)
            H_test = []
            for data_test in test_loader:
                data_test = data_test.to(device)
                self.model(data_test.x, data_test.edge_index, data_test.batch)
                H_test.append(self.H.copy())
            H_test = np.concatenate(H_test, axis=0)
            H_proj = encoder.transform(H_test)
            C_pred = classifier.predict(H_proj)
            accuracies[test_set.concept_names()[concept_id]] = accuracy_score(
                C_test, C_pred
            )
        return accuracies


class GraphCAV(GraphConceptExplainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: ConceptDataset,
        layer: nn.Module,
        n_classes: int,
        **kwargs
    ):
        super(GraphCAV, self).__init__(model, dataset, layer)
        self.n_classes = n_classes

    def fit(
        self, device: torch.device, concept_set_size: int = 100, batch_size: int = 200
    ) -> None:
        classifiers = []
        for concept_id, concept_name in enumerate(self.dataset.concept_names()):
            classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
            dataset_train, C_train = self.dataset.generate_concept_dataset(
                concept_id, concept_set_size
            )
            train_loader = GraphDataLoader(dataset_train, batch_size, shuffle=False)
            H_train = []
            for data_train in train_loader:
                data_train = data_train.to(device)
                self.model(data_train.x, data_train.edge_index, data_train.batch)
                H_train.append(self.H.copy())
            H_train = np.concatenate(H_train, axis=0)
            classifier.fit(H_train, C_train.numpy())
            classifiers.append(classifier)
        self.classifiers = classifiers

    def forward(self, data: GraphData) -> torch.Tensor:
        assert self.classifiers
        self.model(data.x, data.edge_index, data.batch)
        C_pred = torch.zeros((1, len(self.classifiers)))
        for concept_id, classifier in enumerate(self.classifiers):
            C_pred[:, concept_id] = torch.from_numpy(classifier.predict(self.H))
        return C_pred

    def get_activation_vectors(self):
        assert self.classifiers
        cavs = []
        for classifier in self.classifiers:
            cavs.append(torch.tensor(classifier.coef_).float().reshape(1, -1))
        return torch.cat(cavs, dim=0)

    def concept_accuracy(
        self,
        test_set: ConceptDataset,
        device: torch.device,
        concept_set_size: int = 1000,
        batch_size: int = 200,
    ) -> dict:
        accuracies = {}
        for concept_id, classifier in enumerate(self.classifiers):
            dataset_test, C_test = test_set.generate_concept_dataset(
                concept_id, concept_set_size
            )
            test_loader = GraphDataLoader(dataset_test, batch_size, shuffle=False)
            H_test = []
            for data_test in test_loader:
                data_test = data_test.to(device)
                self.model(data_test.x, data_test.edge_index, data_test.batch)
                H_test.append(self.H.copy())
            H_test = np.concatenate(H_test, axis=0)
            C_pred = classifier.predict(H_test)
            accuracies[test_set.concept_names()[concept_id]] = accuracy_score(
                C_test, C_pred
            )
        return accuracies
