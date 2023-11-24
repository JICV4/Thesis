import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from maldi_zsl.blocks import ResidualBlock, GlobalPool, Permute


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs=6000,
        n_outputs=512,
        layer_dims=[512, 512, 512],
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()

        c = n_inputs
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, n_outputs))

        self.net = nn.Sequential(*layers)

        self.hsize = n_outputs

    def forward(self, spectrum):
        return self.net(spectrum["intensity"])


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size = 5,
        n_outputs = 64,
        hidden_sizes = [16, 32, 64, 128],
        blocks_per_stage = 2,
        kernel_size = 7,
        dropout = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Embedding(vocab_size, hidden_sizes[0]),
            Permute(0,2,1),
        ]
        for i in range(len(hidden_sizes)):
            hdim = hidden_sizes[i]
            for _ in range(blocks_per_stage):
                layers.append(ResidualBlock(hidden_dim = hdim, kernel_size = kernel_size, dropout = dropout))
            
            if hdim != hidden_sizes[-1]:
                layers.append(nn.Conv1d(hdim, hidden_sizes[i+1], kernel_size = 2, stride = 2))
            else:
                layers.append(GlobalPool(pooled_axis = 2, mode = "max"))
        
        layers.append(nn.Linear(hidden_sizes[-1], n_outputs))
        self.net = nn.Sequential(*layers)
        self.hsize = n_outputs

    def forward(self, dna):
        # dna should be [B, L], with B=number of samples and L=sequence length, 
        # and should be filled with integers from 0 to 4, encoding the DNA sequence.
        return self.net(dna) # output will be [B, n_outputs]


class SpeciesClassifier(LightningModule):
    def __init__(
        self,
        mlp_kwargs={},
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.spectrum_embedder = MLP(**mlp_kwargs)
        n_classes = mlp_kwargs["n_outputs"]

        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        return self.spectrum_embedder(batch)

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        loss = F.cross_entropy(logits, batch["species"])

        self.log("train_loss", loss, batch_size=len(batch["species"]))
        return loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        loss = F.cross_entropy(logits, batch["species"])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

        self.accuracy(logits, batch["species"])
        self.log(
            "val_acc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )
        self.top5_accuracy(logits, batch["species"])
        self.log(
            "val_top5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["species"].to(logits).unsqueeze(-1), logits], -1),
            batch["0/loc"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [("total", total_params)]
        return params_per_layer