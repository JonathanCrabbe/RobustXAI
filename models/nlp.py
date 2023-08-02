from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BOWClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        hidden1: int = 100,
        hidden2: int = 50,
        num_labels: int = 2,
    ):
        super(BOWClassifier, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_labels)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        # Compute bow
        x = torch.sum(x, dim=1)
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = torch.count_nonzero(torch.argmax(pred, dim=-1) == y) / len(y)
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = torch.count_nonzero(torch.argmax(pred, dim=-1) == y) / len(y)
        self.log_dict({"test/loss": loss, "test/acc": acc})

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = torch.count_nonzero(y == torch.argmax(pred, dim=-1)) / len(y)
        self.log_dict({"validation/loss": loss, "validation/acc": acc}, prog_bar=True)

    def last_layer(self) -> nn.Module:
        return self.fc3
