import os
import logging
import torch
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from torch.utils.data import Dataset, SubsetRandomSampler
from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod
from torch_geometric.datasets import TUDataset
from utils.misc import to_molecule


class ConceptDataset(ABC, Dataset):

    @property
    @abstractmethod
    def concept_names(self):
        ...

    @abstractmethod
    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        ...


class ECGDataset(ConceptDataset):

    def __init__(self, data_dir: Path, train: bool, balance_dataset: bool,
                 random_seed: int = 42, binarize_label: bool = True):
        """
        Generate a ECG dataset
        Args:
            data_dir: directory where the dataset should be stored
            train: True if the training set should be returned, False for the testing set
            balance_dataset: True if the classes should be balanced with SMOTE
            random_seed: random seed for reproducibility
            binarize_label: True if the label should be binarized (0: normal heartbeat, 1: abnormal heartbeat)
        """
        self.data_dir = data_dir
        if not data_dir.exists():
            os.makedirs(data_dir)
            self.download()
        # Read CSV; extract features and labels
        file_path = data_dir / "mitbih_train.csv" if train else data_dir / "mitbih_test.csv"
        df = pd.read_csv(file_path)
        X = df.iloc[:, :187].values
        y = df.iloc[:, 187].values
        if balance_dataset:
            n_normal = np.count_nonzero(y == 0)
            balancing_dic = {0: n_normal, 1: int(n_normal / 4), 2: int(n_normal / 4),
                             3: int(n_normal / 4), 4: int(n_normal / 4)}
            smote = SMOTE(random_state=random_seed, sampling_strategy=balancing_dic)
            X, y = smote.fit_resample(X, y)
        if binarize_label:
            y = np.where(y >= 1, 1, 0)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.binarize_label = binarize_label

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def download(self) -> None:
        import kaggle
        logging.info(f"Downloading ECG dataset in {self.data_dir}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('shayanfazeli/heartbeat', path=self.data_dir, unzip=True)
        logging.info(f"ECG dataset downloaded in {self.data_dir}")

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        """
        Return a concept dataset with positive/negatives for ECG
        Args:
            random_seed: random seed for reproducibility
            concept_set_size: size of the positive and negative subset
        Returns:
            a concept dataset of the form X (features),C (concept labels)
        """
        assert not self.binarize_label
        mask = self.y == concept_id + 1
        positive_idx = torch.nonzero(mask).flatten()
        negative_idx = torch.nonzero(~mask).flatten()
        positive_loader = torch.utils.data.DataLoader(self, batch_size=concept_set_size, sampler=SubsetRandomSampler(positive_idx))
        negative_loader = torch.utils.data.DataLoader(self, batch_size=concept_set_size, sampler=SubsetRandomSampler(negative_idx))
        X_pos, C_pos = next(iter(positive_loader))
        X_neg, C_neg = next(iter(negative_loader))
        X = torch.concatenate((X_pos, X_neg), 0)
        C = torch.concatenate((torch.ones(concept_set_size), torch.zeros(concept_set_size)), 0)
        rand_perm = torch.randperm(len(X))
        return X[rand_perm], C[rand_perm]

    def concept_names(self):
        return ["Supraventricular", "Premature Ventricular", "Fusion Beats", "Unknown"]


class MutagenicityDataset(ConceptDataset, Dataset):

    def __init__(self, data_dir: Path, train: bool, random_seed: int = 11):
        torch.manual_seed(random_seed)
        dataset = TUDataset(str(data_dir), name='Mutagenicity').shuffle()
        self.dataset = dataset[len(dataset) // 10:] if train else dataset[:len(dataset) // 10]
        self.random_seed = random_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.get(idx)

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        for graph in iter(self.dataset):
            molecule = to_molecule(graph).to_undirected()
            atoms = nx.get_node_attributes(molecule, 'name')
            for node1 in molecule.nodes:
                for node2 in molecule.adj[node1]:
                    print(atoms[node1], atoms[node2])
            break

    def concept_names(self):
        ...
