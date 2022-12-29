import torch
import torch.nn as nn
from captum.attr import Attribution, GradientShap, IntegratedGradients, Saliency, Occlusion, FeatureAblation, FeaturePermutation
from torch_geometric.data import Data as GraphData


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
        if isinstance(self.attr_method, GraphFeatureAblation):
            return self.attr_method(data)
        return self.attr_method.attribute(x, target=data.y, additional_forward_args=(data.edge_index, data.batch))


class GraphFeatureAblation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data: GraphData):
        num_nodes = data.num_nodes
        attribution = torch.zeros(data.x.shape)
        y = self.model(data.x, data.edge_index, data.batch)
        for node in range(num_nodes):
            feature_index = torch.argmax(data.x[node])
            new_data = data.clone()
            new_data.x[node, feature_index] = 0
            y_pert = self.model(new_data.x, new_data.edge_index, new_data.batch)
            attribution[node, feature_index] = (y-y_pert)[0, data.y]
        return attribution


