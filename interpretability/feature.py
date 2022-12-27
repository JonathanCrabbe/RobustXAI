import torch
import torch.nn as nn
from captum.attr import Attribution, GradientShap, IntegratedGradients, Saliency, Occlusion, FeatureAblation, FeaturePermutation


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

    def forward_graph(self, data):
        x = data.x
        if isinstance(self.attr_method, (GradientShap, IntegratedGradients, Saliency)):
            x.requires_grad_()
        if isinstance(self.attr_method, GradientShap):
            baseline = torch.zeros(x.shape).to(x.device)
            return self.attr_method.attribute(x, target=data.y, baselines=baseline, n_samples=1,
                                              additional_forward_args=(data.edge_index, data.batch))
        if isinstance(self.attr_method, IntegratedGradients):
            return self.attr_method.attribute(x, target=data.y, internal_batch_size=x.shape[0],
                                              additional_forward_args=(data.edge_index, data.batch))
        if isinstance(self.attr_method, FeatureAblation):
            model_forward = self.attr_method.forward_func
            def mock_forward(x, edge_index, batch): # Captum requires the first index to be a batch index
                return model_forward(x.squeeze(0), edge_index, batch)
            self.attr_method.forward_func = mock_forward
            return self.attr_method.attribute(x.unsqueeze(0), target=data.y,
                                              additional_forward_args=(data.edge_index, data.batch)).squeeze(0)
        return self.attr_method.attribute(x, target=data.y, additional_forward_args=(data.edge_index, data.batch))

