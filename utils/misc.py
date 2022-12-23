import torch
import random
import numpy as np


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