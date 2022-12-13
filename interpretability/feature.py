import torch
import torch.nn as nn
from captum.attr import Attribution, IntegratedGradients


class FeatureImportance(nn.Module):
    def __init__(self, attr_method: Attribution):
        super().__init__()
        self.attr_method = attr_method

    def forward(self, x, y):
        return self.attr_method.attribute(x, target=y)
