import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, ctc_loss
import numpy as np
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // ratio, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // ratio, channels, kernel_size=1)

    def forward(self, x):
        squeeze = F.avg_pool1d(x, x.size(2)).view(-1, x.size(1), 1)
        excitation = F.gelu(self.fc1(squeeze))
        excitation = self.fc2(excitation)
        return x * F.sigmoid(excitation)

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, padding=0, separable=False):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.gelu = nn.GELU()
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        residual = x if self.residual else None
        x = self.conv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.gelu(x)
        if self.residual:
            padding = residual.size(2) - x.size(2)
            x = F.pad(x, (0, padding))
            x += residual
        return x    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        # C1
        self.layers.append(DilatedConvBlock(1, 344, kernel_size=9, dilation=1, stride=3, padding=3))

        # B1
        self.layers.append(DilatedConvBlock(344, 424, kernel_size=115, dilation=1, stride=1, padding=57, separable=True))
        self.layers.append(DilatedConvBlock(424, 424, kernel_size=115, dilation=1, stride=1, padding=57, separable=True))

        # B2
        self.layers.append(DilatedConvBlock(424, 464, kernel_size=5, dilation=2, stride=1, padding=2, separable=True))
        self.layers.append(DilatedConvBlock(464, 464, kernel_size=5, dilation=4, stride=1, padding=4, separable=True))
        self.layers.append(DilatedConvBlock(464, 464, kernel_size=5, dilation=8, stride=1, padding=8, separable=True))

        # B3
        self.layers.append(DilatedConvBlock(464, 456, kernel_size=123, dilation=1, stride=1, padding=61, separable=True))
        self.layers.append(DilatedConvBlock(456, 456, kernel_size=123, dilation=2, stride=1, padding=122, separable=True))

        # B4
        self.layers.append(DilatedConvBlock(456, 440, kernel_size=9, dilation=2, stride=1, padding=4, separable=True))
        self.layers.append(DilatedConvBlock(440, 440, kernel_size=9, dilation=4, stride=1, padding=8, separable=True))

        # B5
        self.layers.append(DilatedConvBlock(440, 280, kernel_size=31, dilation=2, stride=1, padding=15, separable=True))

        # C2
        self.layers.append(DilatedConvBlock(280, 384, kernel_size=67, dilation=1, stride=1, padding=33))

        # C3
        self.layers.append(DilatedConvBlock(384, 48, kernel_size=15, dilation=1, stride=1, padding=7))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(features, classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = torch.permute(self.layers(x), (2, 0, 1))
        return F.log_softmax(x, dim=-1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.alphabet = ["N", "A", "C", "G", "T"]
        self.encoder = Encoder()
        self.decoder = Decoder(48, len(self.alphabet))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        T, N, C = log_probs.shape
        weights = weights or torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)])
        log_probs_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int64)
        loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'total_loss': loss + label_smoothing_loss, 'loss': loss, 'label_smooth_loss': label_smoothing_loss}

    def loss(self, log_probs, targets, lengths):
        return self.ctc_label_smoothing_loss(log_probs, targets, lengths)
    
class SqueezeExcitationModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.model = Rodan()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        event, event_len, label, label_len = batch
        event = torch.unsqueeze(event, 1)
        label = label[:, :max(label_len)]
        output = self(event)
        loss = ctc_loss(output, label, event_len, label_len, reduction="mean", blank=0, zero_infinity=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        wandb.log({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            event, event_len, label, label_len = batch
            event = torch.unsqueeze(event, 1)
            label = label[:, :max(label_len)]
            output = self(event)
            loss = ctc_loss(output, label, event_len, label_len, reduction="mean", blank=0, zero_infinity=True)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            wandb.log({"val_loss": loss})
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    

if __name__ == "__main__":
    model = Model()
    print(model)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))  

    input = torch.randn(32, 1, 4096)
    output = model(input)
    print(output.shape)