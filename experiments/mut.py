import torch
import os
import logging
from torch_geometric.loader import DataLoader
from datasets.loaders import MutagenicityDataset
from models.graphs import ClassifierMutagenicity
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
latent_dim = 32
data_dir = Path.cwd()/'datasets/mut'
model_name = 'model'
model_dir = Path.cwd()/f'results/mut/{model_name}'
if not model_dir.exists():
    os.makedirs(model_dir)
train_set = MutagenicityDataset(data_dir, train=True)
test_set = MutagenicityDataset(data_dir, train=False)
train_loader = DataLoader(train_set, batch_size=500)
test_loader = DataLoader(test_set, batch_size=1)
model = ClassifierMutagenicity(latent_dim)
model.fit(device, train_loader, test_loader, model_dir)

