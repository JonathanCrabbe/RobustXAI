import torch
import torch.nn as nn
from captum.attr import Attribution, GradientShap, IntegratedGradients


class FeatureImportance(nn.Module):
    def __init__(self, attr_method: Attribution):
        super().__init__()
        if isinstance(attr_method, (GradientShap, IntegratedGradients)):
            attr_method._multiply_by_inputs = False
        self.attr_method = attr_method

    def forward(self, x, y):
        if isinstance(self.attr_method, GradientShap):
            baseline = torch.zeros(x.shape).to(x.device)
            return self.attr_method.attribute(x, target=y, baselines=baseline)
        else:
            return self.attr_method.attribute(x, target=y)
