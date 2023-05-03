import torch
import random
import numpy as np
import os
import glob
from networkx import Graph
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_networkx
from typing import List
from pathlib import Path


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def direct_sum(input_tensors):
    """
    Takes a list of tensors and stacks them into one tensor
    """
    unrolled = [tensor.flatten() for tensor in input_tensors]
    return torch.cat(unrolled)


def to_molecule(data: GraphData) -> Graph:
    ATOM_MAP = [
        "C",
        "O",
        "Cl",
        "H",
        "N",
        "F",
        "Br",
        "S",
        "P",
        "I",
        "Na",
        "K",
        "Li",
        "Ca",
    ]
    g = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
    for u, data in g.nodes(data=True):
        data["name"] = ATOM_MAP[data["x"].index(1.0)]
        del data["x"]
    for u, v, data in g.edges(data=True):
        data["valence"] = data["edge_attr"].index(1.0) + 1
        del data["edge_attr"]
    return g


def get_all_cherckpoint_paths(checkpoint_dir: Path) -> List[str]:
    """
    Returns the list of all checkpoints in the given directory
    """
    return glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))


def get_best_checkpoint(checkpoint_dir: Path) -> str:
    """
    Returns the path to the checkpoint with the highest validation accuracy
    """
    checkpoint_paths = get_all_cherckpoint_paths(checkpoint_dir)
    accuracies = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        accuracies.append(checkpoint["val/acc"])
    best_checkpoint_idx = np.argmax(accuracies)
    return checkpoint_paths[best_checkpoint_idx]
