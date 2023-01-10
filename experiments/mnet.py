import logging
from datasets.loaders import ModelNet40Dataset
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data_path = Path.cwd()/'datasets/mnet'
    train_set = ModelNet40Dataset(data_path, train=True, random_seed=42)