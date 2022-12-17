import torch
import torch.nn as nn
from captum.attr import Attribution, GradientShap, IntegratedGradients, Saliency, Occlusion


class FeatureImportance(nn.Module):
    def __init__(self, attr_method: Attribution):
        super().__init__()
        self.attr_method = attr_method

    def forward(self, x, y):
        if isinstance(self.attr_method, (GradientShap, IntegratedGradients, Saliency)):
            x.requires_grad_()
        if isinstance(self.attr_method, GradientShap):
            baseline = torch.zeros(x.shape).to(x.device)
            return self.attr_method.attribute(x, target=y, baselines=baseline)
        if isinstance(self.attr_method, IntegratedGradients):
            return self.attr_method.attribute(x, target=y, internal_batch_size=len(x))
        if isinstance(self.attr_method, Occlusion):
            windows_shapes = (1,) + (len(x.shape)-2)*(5,)
            return self.attr_method.attribute(x, target=y, sliding_window_shapes=windows_shapes)
        else:
            return self.attr_method.attribute(x, target=y)
