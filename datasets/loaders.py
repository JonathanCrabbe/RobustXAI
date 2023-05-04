import os
import logging
import torch
import pandas as pd
import numpy as np
import networkx as nx
import random
import h5py
import pytorch_lightning as pl
from torchvision.transforms import transforms
from pathlib import Path
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision.datasets import FashionMNIST, CIFAR100, STL10
from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod
from torch_geometric.datasets import TUDataset
from utils.misc import to_molecule
from joblib import Parallel, delayed
from tqdm import tqdm


class ConceptDataset(ABC, Dataset):
    @property
    @abstractmethod
    def concept_names(self):
        ...

    @abstractmethod
    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        ...


class ECGDataset(ConceptDataset):
    def __init__(
        self,
        data_dir: Path,
        train: bool,
        balance_dataset: bool,
        random_seed: int = 42,
        binarize_label: bool = True,
    ):
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
        file_path = (
            data_dir / "mitbih_train.csv" if train else data_dir / "mitbih_test.csv"
        )
        df = pd.read_csv(file_path)
        X = df.iloc[:, :187].values
        y = df.iloc[:, 187].values
        if balance_dataset:
            n_normal = np.count_nonzero(y == 0)
            balancing_dic = {
                0: n_normal,
                1: int(n_normal / 4),
                2: int(n_normal / 4),
                3: int(n_normal / 4),
                4: int(n_normal / 4),
            }
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
        kaggle.api.dataset_download_files(
            "shayanfazeli/heartbeat", path=self.data_dir, unzip=True
        )
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
        positive_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(positive_idx)
        )
        negative_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(negative_idx)
        )
        X_pos, C_pos = next(iter(positive_loader))
        X_neg, C_neg = next(iter(negative_loader))
        X = torch.concatenate((X_pos, X_neg), 0)
        C = torch.concatenate(
            (torch.ones(concept_set_size), torch.zeros(concept_set_size)), 0
        )
        rand_perm = torch.randperm(len(X))
        return X[rand_perm], C[rand_perm]

    def concept_names(self):
        return ["Supraventricular", "Premature Ventricular", "Fusion Beats", "Unknown"]


class MutagenicityDataset(ConceptDataset, Dataset):
    def __init__(self, data_dir: Path, train: bool, random_seed: int = 11):
        torch.manual_seed(random_seed)
        dataset = TUDataset(str(data_dir), name="Mutagenicity").shuffle()
        self.dataset = (
            dataset[len(dataset) // 10 :] if train else dataset[: len(dataset) // 10]
        )
        self.random_seed = random_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.get(idx)

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        concept_detectors = {
            "Nitroso": self.is_nitroso,
            "Aliphatic Halide": self.is_aliphatic_halide,
            "Azo Type": self.is_azo_type,
            "Nitro Type": self.is_nitro_type,
        }
        concept_detector = concept_detectors[self.concept_names()[concept_id]]
        mask = []
        for graph in iter(self.dataset):
            molecule = to_molecule(graph)
            mask.append(concept_detector(molecule))
        mask = torch.tensor(mask)
        positive_set = self.dataset[mask][:concept_set_size]
        negative_set = self.dataset[~mask][:concept_set_size]
        concept_set = positive_set + negative_set
        C = torch.concatenate(
            (torch.ones(len(positive_set)), torch.zeros(len(negative_set))), 0
        )
        return concept_set, C

    def concept_names(self):
        return ["Nitroso", "Aliphatic Halide", "Azo Type", "Nitro Type"]

    @staticmethod
    def is_nitroso(molecule: nx.Graph) -> bool:
        atoms = nx.get_node_attributes(molecule, "name")
        valences = nx.get_edge_attributes(molecule, "valence")
        for node1 in molecule.nodes:
            if atoms[node1] == "N":
                for node2 in molecule.adj[node1]:
                    if atoms[node2] == "O" and valences[node1, node2] == 2:
                        return True
        return False

    @staticmethod
    def is_aliphatic_halide(molecule: nx.Graph) -> bool:
        atoms = nx.get_node_attributes(molecule, "name")
        for node1 in molecule.nodes:
            if atoms[node1] in {"Cl", "Br", "I"}:
                return True
        return False

    @staticmethod
    def is_azo_type(molecule: nx.Graph) -> bool:
        atoms = nx.get_node_attributes(molecule, "name")
        valences = nx.get_edge_attributes(molecule, "valence")
        for node1 in molecule.nodes:
            if atoms[node1] == "N":
                for node2 in nx.neighbors(molecule, node1):
                    if atoms[node2] == "N" and valences[node1, node2] == 2:
                        return True
        return False

    @staticmethod
    def is_nitro_type(molecule: nx.Graph) -> bool:
        atoms = nx.get_node_attributes(molecule, "name")
        valences = nx.get_edge_attributes(molecule, "valence")
        for node1 in molecule.nodes:
            if atoms[node1] == "N":
                has_single_NO = False
                has_double_NO = False
                for node2 in nx.neighbors(molecule, node1):
                    if atoms[node2] == "O":
                        match valences[node1, node2]:
                            case 1:
                                has_single_NO = True
                            case 2:
                                has_double_NO = True

                if has_single_NO and has_double_NO:
                    return True
        return False


class ModelNet40Dataset(ConceptDataset):
    def __init__(
        self,
        data_dir: Path,
        train: bool,
        random_seed: int = 42,
        down_sample=10,
        do_standardize=True,
    ):
        """
        Generate a ModelNet40 dataset
        Args:
            data_dir: directory where the dataset should be stored
            train: True if the training set should be returned, False for the testing set
            random_seed: random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_seed = random_seed
        if not data_dir.exists():
            os.makedirs(data_dir)
            self.download()
        if not (self.data_dir / "ModelNet_40_npy").exists():
            self.preprocess()
        if not (self.data_dir / "ModelNet40_cloud.h5").exists():
            self.formatting()

        self.down_sample = down_sample
        with h5py.File(self.data_dir / "ModelNet40_cloud.h5", "r") as f:
            if train:
                self.X = np.array(f["tr_cloud"])
                self.Y = np.array(f["tr_label"])
            else:
                self.X = np.array(f["test_cloud"])
                self.Y = np.array(f["test_label"])

        self.num_classes = np.max(self.Y) + 1
        self.prep = self.standardize if do_standardize else lambda x: x

        # Select the subset of points to use throughout beforehand
        np.random.seed(random_seed)
        self.perm = np.random.permutation(self.X.shape[1])[:: self.down_sample]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.prep(self.X[idx, self.perm])
        y = self.Y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def download(self) -> None:
        import kaggle

        logging.info(f"Downloading ModelNet40 dataset in {self.data_dir}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "balraj98/modelnet40-princeton-3d-object-dataset",
            path=self.data_dir,
            unzip=True,
        )
        logging.info(f"ECG dataset downloaded in {self.data_dir}")

    def preprocess(self) -> None:
        """
        Preprocessing code adapted from https://github.com/michaelsdr/sinkformers/blob/main/model_net_40/to_h5.py
        :return:
        """
        random.seed(self.random_seed)
        path = self.data_dir / "ModelNet40"
        folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
        classes = {folder: i for i, folder in enumerate(folders)}

        def read_off(file):
            header = file.readline().strip()
            if "OFF" not in header:
                raise ValueError("Not a valid OFF header")
            if header != "OFF":  # The header is merged with the first line
                second_line = header.replace("OFF", "")
            else:  # The second line can be read directly
                second_line = file.readline()
            n_verts, n_faces, __ = tuple(
                [int(s) for s in second_line.strip().split(" ")]
            )
            verts = [
                [float(s) for s in file.readline().strip().split(" ")]
                for i_vert in range(n_verts)
            ]
            faces = [
                [int(s) for s in file.readline().strip().split(" ")][1:]
                for i_face in range(n_faces)
            ]
            return verts, faces

        with open(path / "bed/train/bed_0001.off", "r") as f:
            verts, faces = read_off(f)

        i, j, k = np.array(faces).T
        x, y, z = np.array(verts).T

        class PointSampler(object):
            def __init__(self, output_size):
                assert isinstance(output_size, int)
                self.output_size = output_size

            def triangle_area(self, pt1, pt2, pt3):
                side_a = np.linalg.norm(pt1 - pt2)
                side_b = np.linalg.norm(pt2 - pt3)
                side_c = np.linalg.norm(pt3 - pt1)
                s = 0.5 * (side_a + side_b + side_c)
                return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

            def sample_point(self, pt1, pt2, pt3):
                # barycentric coordinates on a triangle
                # https://mathworld.wolfram.com/BarycentricCoordinates.html
                s, t = sorted([random.random(), random.random()])
                f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
                return (f(0), f(1), f(2))

            def __call__(self, mesh):
                verts, faces = mesh
                verts = np.array(verts)
                areas = np.zeros((len(faces)))

                for i in range(len(areas)):
                    areas[i] = self.triangle_area(
                        verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
                    )

                sampled_faces = random.choices(
                    faces, weights=areas, cum_weights=None, k=self.output_size
                )

                sampled_points = np.zeros((self.output_size, 3))

                for i in range(len(sampled_faces)):
                    sampled_points[i] = self.sample_point(
                        verts[sampled_faces[i][0]],
                        verts[sampled_faces[i][1]],
                        verts[sampled_faces[i][2]],
                    )

                return sampled_points

        pointcloud = PointSampler(10000)((verts, faces))

        def process(file, file_adr, save_adr):
            fname = save_adr / f"{file[:-4]}.npy"
            if file_adr.suffix == ".off":
                if not os.path.isfile(fname):
                    with open(file_adr, "r") as f:
                        verts, faces = read_off(f)
                        pointcloud = PointSampler(10000)((verts, faces))
                        np.save(fname, pointcloud)
                else:
                    pass

        tr_label = []
        tr_cloud = []
        test_cloud = []
        test_label = []

        folder = "train"
        root_dir = path
        folders = [
            dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)
        ]
        classes = {folder: i for i, folder in enumerate(folders)}
        files = []

        all_files = []
        all_files_adr = []
        all_save_adr = []

        for category in classes.keys():
            save_adr = self.data_dir / f"ModelNet_40_npy/{category}/{folder}"
            try:
                os.makedirs(save_adr)
            except:
                pass
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                all_files.append(file)
                all_files_adr.append(new_dir / file)
                all_save_adr.append(save_adr)

        logging.info("Now processing the training files")
        Parallel(n_jobs=40)(
            delayed(process)(file, file_adr, save_adr)
            for (file, file_adr, save_adr) in tqdm(
                zip(all_files, all_files_adr, all_save_adr), leave=False, unit="files"
            )
        )

        folder = "test"
        root_dir = path
        folders = [
            dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)
        ]
        classes = {folder: i for i, folder in enumerate(folders)}
        files = []

        all_files = []
        all_files_adr = []
        all_save_adr = []

        for category in classes.keys():
            save_adr = self.data_dir / f"ModelNet_40_npy/{category}/{folder}"
            try:
                os.makedirs(save_adr)
            except:
                pass
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                all_files.append(file)
                all_files_adr.append(new_dir / file)
                all_save_adr.append(save_adr)

        logging.info("Now processing the test files")
        Parallel(n_jobs=40)(
            delayed(process)(file, file_adr, save_adr)
            for (file, file_adr, save_adr) in tqdm(
                zip(all_files, all_files_adr, all_save_adr), leave=False, unit="files"
            )
        )

    def formatting(self) -> None:
        """
        Preprocessing code adapted from https://github.com/michaelsdr/sinkformers/blob/main/model_net_40/formatting.py
        :return:
        """
        path = self.data_dir / "ModelNet_40_npy"

        folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
        classes = {folder: i for i, folder in enumerate(folders)}

        tr_label = []
        tr_cloud = []
        test_cloud = []
        test_label = []

        logging.info("Now formatting training files")
        folder = "train"
        root_dir = path
        folders = [
            dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)
        ]
        classes = {folder: i for i, folder in enumerate(folders)}
        files = []

        all_files = []
        all_files_adr = []
        all_save_adr = []

        for category, num in zip(classes.keys(), classes.values()):
            new_dir = root_dir / Path(category) / folder

            for file in os.listdir(new_dir):
                if file.endswith(".npy"):
                    try:
                        point_cloud = np.load(new_dir / file)
                        tr_cloud.append(point_cloud)
                        tr_label.append(num)
                    except:
                        pass
        tr_cloud = np.asarray(tr_cloud)
        tr_label = np.asarray(tr_label)
        np.save(str(self.data_dir / "tr_cloud.npy"), tr_cloud)
        np.save(str(self.data_dir / "tr_label.npy"), tr_label)

        logging.info("Now formatting test files")
        folder = "test"
        root_dir = path
        folders = [
            dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)
        ]
        classes = {folder: i for i, folder in enumerate(folders)}
        files = []

        for category, num in zip(classes.keys(), classes.values()):
            new_dir = root_dir / Path(category) / folder

            for file in os.listdir(new_dir):
                if file.endswith(".npy"):
                    try:
                        point_cloud = np.load(new_dir / file)
                        test_cloud.append(point_cloud)
                        test_label.append(num)
                    except:
                        pass

        test_cloud = np.asarray(test_cloud)
        test_label = np.asarray(test_label)
        np.save(str(self.data_dir / "test_cloud.npy"), test_cloud)
        np.save(str(self.data_dir / "test_label.npy"), test_label)

        with h5py.File(self.data_dir / "ModelNet40_cloud.h5", "w") as f:
            f.create_dataset("test_cloud", data=test_cloud)
            f.create_dataset("tr_cloud", data=tr_cloud)
            f.create_dataset("test_label", data=test_label)
            f.create_dataset("tr_label", data=tr_label)

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        """
        Return a concept dataset with positive/negatives for ModelNet40
        Args:
            random_seed: random seed for reproducibility
            concept_set_size: size of the positive and negative subset
        Returns:
            a concept dataset of the form X (features),C (concept labels)
        """
        path = self.data_dir / "ModelNet_40_npy"
        classes = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
        concept_to_classes = {
            "Foot": {
                "bed",
                "bench",
                "chair",
                "desk",
                "lamp",
                "person",
                "piano",
                "sofa",
                "stool",
                "table",
            },
            "Container": {
                "bathtub",
                "bottle",
                "bowl",
                "cup",
                "flower_pot",
                "sink",
                "toilet",
                "vase",
            },
            "Parallelepiped": {
                "bed",
                "bookshelf",
                "desk",
                "door",
                "dresser",
                "glass_box",
                "monitor",
                "night_stand",
                "radio",
                "sofa",
                "tv_stand",
                "wardrobe",
                "xbox",
            },
            "Elongated": {
                "airplane",
                "bathtub",
                "bed",
                "bench",
                "bottle",
                "car",
                "chair",
                "desk",
                "door",
                "guitar",
                "lamp",
                "person",
                "sofa",
                "stairs",
                "table",
            },
        }
        mask = []
        for y in self.Y:
            mask.append(
                classes[y] in concept_to_classes[self.concept_names()[concept_id]]
            )
        mask = torch.tensor(mask)
        positive_idx = torch.nonzero(mask).flatten()
        negative_idx = torch.nonzero(~mask).flatten()
        positive_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(positive_idx)
        )
        negative_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(negative_idx)
        )
        X_pos, _ = next(iter(positive_loader))
        X_neg, _ = next(iter(negative_loader))
        X = torch.concatenate((X_pos, X_neg), 0)
        C = torch.concatenate(
            (torch.ones(concept_set_size), torch.zeros(concept_set_size)), 0
        )
        rand_perm = torch.randperm(len(X))
        return X[rand_perm], C[rand_perm]

    def concept_names(self):
        return ["Foot", "Container", "Parallelepiped", "Elongated"]

    @staticmethod
    def standardize(x):
        clipper = np.mean(np.abs(x), (0, 1), keepdims=True)
        z = np.clip(x, -100 * clipper, 100 * clipper)
        mean = np.mean(z, (0, 1), keepdims=True)
        std = np.std(z, (0, 1), keepdims=True)
        return (z - mean) / std


class FashionMnistDataset(ConceptDataset, FashionMNIST):
    def __init__(self, data_dir: Path, train: bool, max_displacement: int) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad(max_displacement)]
        )
        super().__init__(data_dir, train, transform, download=True)

    def concept_names(self):
        return ["Top", "Shoe"]

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        labels = self.targets
        concept_label_sets = {"Top": [0, 2, 3, 4, 6], "Shoe": [5, 7, 9]}
        mask = torch.isin(
            labels, torch.Tensor(concept_label_sets[self.concept_names()[concept_id]])
        )
        positive_idx = torch.nonzero(mask).flatten()
        negative_idx = torch.nonzero(~mask).flatten()
        positive_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(positive_idx)
        )
        negative_loader = torch.utils.data.DataLoader(
            self, batch_size=concept_set_size, sampler=SubsetRandomSampler(negative_idx)
        )
        X_pos, C_pos = next(iter(positive_loader))
        X_neg, C_neg = next(iter(negative_loader))
        X = torch.concatenate((X_pos, X_neg), 0)
        C = torch.concatenate(
            (torch.ones(concept_set_size), torch.zeros(concept_set_size)), 0
        )
        rand_perm = torch.randperm(len(X))
        return X[rand_perm], C[rand_perm]


class Cifar100Dataset(pl.LightningDataModule, ConceptDataset):
    def __init__(self, data_dir: Path, batch_size: int = 32, num_predict: int = 500):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_predict = num_predict

    def setup(self, stage: str):
        normalize = transforms.Normalize(
            mean=np.array([125.3, 123.0, 113.9]) / 255.0,
            std=np.array([63.0, 62.1, 66.7]) / 255.0,
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.cifar100_test = CIFAR100(
            self.data_dir, train=False, download=True, transform=valid_transform
        )
        self.cifar100_train = CIFAR100(
            self.data_dir, train=True, download=True, transform=train_transform
        )
        self.cifar100_val = CIFAR100(
            self.data_dir, train=True, download=True, transform=valid_transform
        )
        num_train = len(self.cifar100_train)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        num_test = len(self.cifar100_test)
        predict_idx = torch.randperm(num_test)[: self.num_predict]
        self.predict_sampler = SubsetRandomSampler(predict_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar100_val, batch_size=self.batch_size, sampler=self.valid_sampler
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar100_test, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar100_test, batch_size=self.batch_size, sampler=self.predict_sampler
        )

    def teardown(self, stage: str):
        ...

    def concept_names(self):
        return ["Aquatic", "People", "Vehicles"]

    def generate_concept_dataset(self, concept_id: int, concept_set_size: int) -> tuple:
        train_set = self.cifar100_train
        classes = train_set.classes
        concept_to_classes = {
            "Aquatic": {
                "beaver",
                "dolphin",
                "otter",
                "seal",
                "whale",
                "aquarium_fish",
                "flatfish",
                "ray",
                "shark",
                "trout",
            },
            "People": {"baby", "boy", "girl", "man", "woman"},
            "Vehicles": {
                "bicycle",
                "bus",
                "motorcycle",
                "pickup_truck",
                "train",
                "lawn_mower",
                "rocket",
                "streetcar",
                "tank",
                "tractor",
            },
        }
        labels = torch.Tensor(self.cifar100_train.targets)
        mask = []
        for label in labels:
            mask.append(
                classes[int(label)]
                in concept_to_classes[self.concept_names()[concept_id]]
            )
        mask = torch.BoolTensor(mask)
        positive_idx = torch.nonzero(mask).flatten()
        negative_idx = torch.nonzero(~mask).flatten()
        positive_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=concept_set_size,
            sampler=SubsetRandomSampler(positive_idx),
        )
        negative_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=concept_set_size,
            sampler=SubsetRandomSampler(negative_idx),
        )
        X_pos, C_pos = next(iter(positive_loader))
        X_neg, C_neg = next(iter(negative_loader))
        X = torch.concatenate((X_pos, X_neg), 0)
        C = torch.concatenate(
            (torch.ones(concept_set_size), torch.zeros(concept_set_size)), 0
        )
        rand_perm = torch.randperm(len(X))
        return X[rand_perm], C[rand_perm]


class STL10Dataset(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32, num_predict: int = 500):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_predict = num_predict

    def setup(self, stage: str):
        mean = np.array([0.44508205, 0.43821473, 0.40541945])
        std = np.array([0.26199411, 0.25827974, 0.27239384])
        normalize = transforms.Normalize(
            mean=mean,
            std=std,
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(96, padding=12),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(60),
                normalize,
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.stl10_test = STL10(
            self.data_dir, split="test", download=True, transform=valid_transform
        )
        self.stl10_train = STL10(
            self.data_dir, split="train", download=True, transform=train_transform
        )
        self.stl10_val = STL10(
            self.data_dir, split="train", download=True, transform=valid_transform
        )
        num_train = len(self.stl10_train)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        num_test = len(self.stl10_test)
        predict_idx = torch.randperm(num_test)[: self.num_predict]
        self.predict_sampler = SubsetRandomSampler(predict_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.stl10_train,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.stl10_val, batch_size=self.batch_size, sampler=self.valid_sampler
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.stl10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.stl10_test, batch_size=self.batch_size, sampler=self.predict_sampler
        )

    def teardown(self, stage: str):
        ...


class Cutout:
    """Randomly mask out a patch from an image.
    Args:
        size (int): The size of the square patch.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image
        Returns:
            Tensor: Image with a hole of dimension size x size cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
