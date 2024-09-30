# #!/usr/bin/env python
# # 
# # RODAN
# # v1.0
# # (c) 2020,2021,2022 Don Neumann
# #

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import h5py
# from torch.autograd import Variable
# import sys, argparse, shutil, pickle, os, copy, math
# from collections import OrderedDict

# import wandb
# import pytorch_lightning as pl

# from utils.loss import ctc_label_smoothing_loss
# from pytorch_lightning import Trainer

# defaultconfig = {"name":"default", "seqlen":4096, "epochs":30, "optimizer":"ranger", "lr":2e-3, "weightdecay":0.01, "batchsize":30, "dropout": 0.1, "activation":"mish", "sqex_activation":"mish", "sqex_reduction":32, "trainfile":"rna-train.hdf5", "validfile":"rna-valid.hdf5", "amp":False, "scheduler":"reducelronplateau", "scheduler_patience":1, "scheduler_factor":0.5, "scheduler_threshold":0.1, "scheduler_minlr": 1e-05, "scheduler_reduce":2, "gradclip":0, "train_loopcount": 1000000, "valid_loopcount": 1000, "tensorboard":False, "saveinit":False,
#         "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}

# rna_default = [[-1, 256, 0, 3, 1, 1, 0], [-1, 256, 1, 10, 1, 1, 1], [-1, 256, 1, 10, 10, 1, 1], [-1, 320, 1, 10, 1, 1, 1], [-1, 384, 1, 15, 1, 1, 1], [-1, 448, 1, 20, 1, 1, 1], [-1, 512, 1, 25, 1, 1, 1], [-1, 512, 1, 30, 1, 1, 1], [-1, 512, 1, 35, 1, 1, 1], [-1, 512, 1, 40, 1, 1, 1], [-1, 512, 1, 45, 1, 1, 1], [-1, 512, 1, 50, 1, 1, 1], [-1, 768, 1, 55, 1, 1, 1], [-1, 768, 1, 60, 1, 1, 1], [-1, 768, 1, 65, 1, 1, 1], [-1, 768, 1, 70, 1, 1, 1], [-1, 768, 1, 75, 1, 1, 1], [-1, 768, 1, 80, 1, 1, 1], [-1, 768, 1, 85, 1, 1, 1], [-1, 768, 1, 90, 1, 1, 1], [-1, 768, 1, 95, 1, 1, 1], [-1, 768, 1, 100, 1, 1, 1]]
# dna_default = [[-1, 320, 0, 3, 1, 1, 0], [-1, 320, 1, 3, 3, 1, 1], [-1, 384, 1, 6, 1, 1, 1], [-1, 448, 1, 9, 1, 1, 1], [-1, 512, 1, 12, 1, 1, 1], [-1, 576, 1, 15, 1, 1, 1], [-1, 640, 1, 18, 1, 1, 1], [-1, 704, 1, 21, 1, 1, 1], [-1, 768, 1, 24, 1, 1, 1], [-1, 832, 1, 27, 1, 1, 1], [-1, 896, 1, 30, 1, 1, 1], [-1, 960, 1, 33, 1, 1, 1]]

# class objectview(object):
#     def __init__(self, d):
#         self.__dict__ = d
#         self.orig = d

# def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
#     nrows = ((a.size-L)//S)+1
#     n = a.strides[0]
#     return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

# class Swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(Swish, self).__init__()
#         self.inplace = inplace

#     def forward(self, x):
#         return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x *( torch.tanh(torch.nn.functional.softplus(x)))

# class squeeze_excite(torch.nn.Module):
#     def __init__(self, in_channels = 512, size=1, reduction="/16", activation=torch.nn.GELU):
#         super(squeeze_excite, self).__init__()
#         self.in_channels = in_channels
#         self.avg = torch.nn.AdaptiveAvgPool1d(1)
#         if type(reduction) == str:
#             self.reductionsize = self.in_channels // int(reduction[1:])
#         else:
#             self.reductionsize = reduction
#         self.fc1 = nn.Linear(self.in_channels, self.reductionsize)
#         self.activation = activation() # was nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(self.reductionsize, self.in_channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         input = x
#         x = self.avg(x)
#         x = x.permute(0,2,1)
#         x = self.activation(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return input * x.permute(0,2,1)


# class convblock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, seperable=True, expansion=True, batchnorm=True, dropout=0.1, activation=torch.nn.GELU, sqex=True, squeeze=32, sqex_activation=torch.nn.GELU, residual=True):
#         # no bias?
#         super(convblock, self).__init__()
#         self.seperable = seperable
#         self.batchnorm = batchnorm
#         self.dropout = dropout
#         self.activation = activation
#         self.squeeze = squeeze
#         self.stride = stride
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.residual = residual
#         self.doexpansion = expansion
#         # fix self.squeeze
#         dwchannels = in_channels
#         if seperable:
#             if self.doexpansion and self.in_channels != self.out_channels:
#                 self.expansion = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False)
#                 self.expansion_norm = torch.nn.BatchNorm1d(out_channels)
#                 self.expansion_act = self.activation()
#                 dwchannels = out_channels 

#             self.depthwise = torch.nn.Conv1d(dwchannels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=out_channels//groups)
#             if self.batchnorm:
#                 self.bn1 = torch.nn.BatchNorm1d(out_channels)
#             self.act1 = self.activation()
#             if self.squeeze:
#                 self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
#             self.pointwise = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=dilation, bias=bias, padding=0)
#             if self.batchnorm:
#                 self.bn2 = torch.nn.BatchNorm1d(out_channels)
#             self.act2 = self.activation()
#             if self.dropout:
#                 self.drop = torch.nn.Dropout(self.dropout)
#         else:
#             self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
#             if self.batchnorm:
#                 self.bn1 = torch.nn.BatchNorm1d(out_channels)
#             self.act1 = self.activation()
#             if self.squeeze:
#                 self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
#             if self.dropout:
#                 self.drop = torch.nn.Dropout(self.dropout)
#         if self.residual and self.stride == 1:
#             self.rezero = nn.Parameter(torch.Tensor([0]), requires_grad=True)

#     def forward(self, x):
#         orig = x

#         if self.seperable:
#             if self.in_channels != self.out_channels and self.doexpansion:
#                 x = self.expansion(x)
#                 x = self.expansion_norm(x)
#                 x = self.expansion_act(x)
#             x = self.depthwise(x)
#             if self.batchnorm: x = self.bn1(x)
#             x = self.act1(x)
#             if self.squeeze:
#                 x = self.sqex(x)
#             x = self.pointwise(x)
#             if self.batchnorm: x = self.bn2(x)
#             x = self.act2(x) 
#             if self.dropout: x = self.drop(x)
#         else:
#             x = self.conv(x)
#             if self.batchnorm: x = self.bn1(x)
#             x = self.act1(x)
#             if self.dropout: x = self.drop(x)

#         if self.residual and self.stride == 1 and self.in_channels == self.out_channels and x.shape[2] == orig.shape[2]:
#             return orig + self.rezero * x # rezero
#             #return orig + x # normal residual
#         else:
#             return x

# def activation_function(activation):
#     if activation == "mish":
#         return Mish
#     elif activation == "swish":
#         return Swish
#     elif activation == "relu":
#         return torch.nn.ReLU
#     elif activation == "gelu":
#         return torch.nn.GELU
#     else:
#         print("Unknown activation type:", activation)
#         sys.exit(1)
    
# class network(nn.Module):
#     def __init__(self, config=defaultconfig, arch=None, seqlen=4096, debug=False):
#         super().__init__()
#         if debug: print("Initializing network")
        
#         self.seqlen = seqlen
#         self.vocab = config["vocab"]
        
#         self.bn = nn.BatchNorm1d

#         # [P, Channels, Separable, kernel_size, stride, sqex, dropout]
#         # P = -1 kernel_size//2, 0 none, >0 used as padding
#         # Channels
#         # seperable = 0 False, 1 True
#         # kernel_size
#         # stride
#         # sqex = 0 False, 1 True
#         # dropout = 0 False, 1 True
#         if arch == None: arch = rna_default

#         activation = activation_function(config["activation"].lower())
#         sqex_activation = activation_function(config["sqex_activation"].lower())

#         self.convlayers = nn.Sequential()
#         in_channels = 1
#         convsize = self.seqlen

#         for i, layer in enumerate(arch):
#             paddingarg = layer[0]
#             out_channels = layer[1]
#             seperable = layer[2] 
#             kernel = layer[3]
#             stride = layer[4]
#             sqex = layer[5]
#             dodropout = layer[6]
#             expansion = True

#             if dodropout: dropout = config["dropout"]
#             else: dropout = 0
#             if sqex: squeeze = config["sqex_reduction"]
#             else: squeeze = 0

#             if paddingarg == -1:
#                 padding = kernel // 2
#             else: padding = paddingarg
#             if i == 0: expansion = False

#             convsize = (convsize + (padding*2) - (kernel-stride))//stride
#             if debug:
#                 print("padding:", padding, "seperable:", seperable, "ch", out_channels, "k:", kernel, "s:", stride, "sqex:", sqex, "drop:", dropout, "expansion:", expansion)
#                 print("convsize:", convsize)
#             self.convlayers.add_module("conv"+str(i), convblock(in_channels, out_channels, kernel, stride=stride, padding=padding, seperable=seperable, activation=activation, expansion=expansion, dropout=dropout, squeeze=squeeze, sqex_activation=sqex_activation, residual=True))
#             in_channels = out_channels
#             self.final_size = out_channels
         
#         self.final = nn.Linear(self.final_size, len(self.vocab))
#         if debug: print("Finished init network")

#     def forward(self, x):
#         #x = self.embedding(x)
#         x = self.convlayers(x)
#         x = x.permute(0,2,1)
#         x = self.final(x)
#         x = torch.nn.functional.log_softmax(x, 2)
#         return x.permute(1, 0, 2)

# counter = 0

# def get_checkpoint(epoch, model, optimizer, scheduler):
#     checkpoint = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict()
#         }
#     return checkpoint

# def get_config(model, config):
#     config = {
#             "state_dict": model.state_dict(),
#             "config": config
#             }
#     return config


# class RodanModule(pl.LightningModule):
#     def __init__(self, lr=1e-3, weight_decay=0.01):
#         super().__init__()
#         self.model = network()
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.save_hyperparameters()
#         self.smoothweights = torch.cat([torch.tensor([0.1]), (0.1 / (5 - 1)) * torch.ones(5 - 1)])

#     def forward(self, x):
#         return self.model(x)
    
#     def get_lr(self):
#         optimizer = self.trainer.optimizers[0]
#         return optimizer.param_groups[0]['lr']

#     def training_step(self, batch, batch_idx):
#         self.model.train()
#         event, event_len, label, label_len = batch
#         event = torch.unsqueeze(event, 1)
#         label = label[:, :max(label_len)]
#         output = self(event)

#         losses = ctc_label_smoothing_loss(output, label, label_len, self.smoothweights)
#         loss = losses["loss"]

#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

#         current_lr = self.get_lr()
#         self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
#         wandb.log({"train_loss": loss,
#                    "learning_rate": current_lr})
        
#         return loss
    
#     def on_train_epoch_end(self):
#         current_lr = self.get_lr()
#         self.log('learning_rate_epoch', current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#         wandb.log({"learning_rate_epoch": current_lr})

#     def validation_step(self, batch, batch_idx):
#         self.model.eval()
#         with torch.no_grad():
#             event, event_len, label, label_len = batch
#             event = torch.unsqueeze(event, 1)
#             label = label[:, :max(label_len)]
#             output = self(event)

#             losses = ctc_label_smoothing_loss(output, label, label_len, self.smoothweights)
#             loss = losses["loss"]            

#             self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
#             wandb.log({"val_loss": loss})
#             return loss

#     def configure_optimizers(self):
#         optimizer = Ranger(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             patience=1,
#             factor=0.5,
#             verbose=False,
#             threshold=0.1,
#             min_lr=1e-05
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",
#             },
#         }

#!/usr/bin/env python
# 
# RODAN
# v1.0
# (c) 2020,2021,2022 Don Neumann
#

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import h5py
from torch.autograd import Variable
import sys, argparse, shutil, pickle, os, copy, math
from collections import OrderedDict
from types import SimpleNamespace

defaultconfig = SimpleNamespace(
    name="default",
    seqlen=4096,
    epochs=30,
    optimizer="ranger",
    lr=2e-3,
    weightdecay=0.01,
    batchsize=30,
    dropout=0.1,
    activation="mish",
    sqex_activation="mish",
    sqex_reduction=32,
    trainfile="rna-train.hdf5",
    validfile="rna-valid.hdf5",
    amp=False,
    scheduler="reducelronplateau",
    scheduler_patience=1,
    scheduler_factor=0.5,
    scheduler_threshold=0.1,
    scheduler_minlr=1e-05,
    scheduler_reduce=2,
    gradclip=0,
    train_loopcount=1000000,
    valid_loopcount=1000,
    tensorboard=False,
    saveinit=False,
    vocab=['<PAD>', 'A', 'C', 'G', 'T']
)

# defaultconfig = {"name":"default", "seqlen":4096, "epochs":30, "optimizer":"ranger", "lr":2e-3, "weightdecay":0.01, "batchsize":30, "dropout": 0.1, "activation":"mish", "sqex_activation":"mish", "sqex_reduction":32, "trainfile":"rna-train.hdf5", "validfile":"rna-valid.hdf5", "amp":False, "scheduler":"reducelronplateau", "scheduler_patience":1, "scheduler_factor":0.5, "scheduler_threshold":0.1, "scheduler_minlr": 1e-05, "scheduler_reduce":2, "gradclip":0, "train_loopcount": 1000000, "valid_loopcount": 1000, "tensorboard":False, "saveinit":False,
#         "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}

rna_default = [[-1, 256, 0, 3, 1, 1, 0], [-1, 256, 1, 10, 1, 1, 1], [-1, 256, 1, 10, 10, 1, 1], [-1, 320, 1, 10, 1, 1, 1], [-1, 384, 1, 15, 1, 1, 1], [-1, 448, 1, 20, 1, 1, 1], [-1, 512, 1, 25, 1, 1, 1], [-1, 512, 1, 30, 1, 1, 1], [-1, 512, 1, 35, 1, 1, 1], [-1, 512, 1, 40, 1, 1, 1], [-1, 512, 1, 45, 1, 1, 1], [-1, 512, 1, 50, 1, 1, 1], [-1, 768, 1, 55, 1, 1, 1], [-1, 768, 1, 60, 1, 1, 1], [-1, 768, 1, 65, 1, 1, 1], [-1, 768, 1, 70, 1, 1, 1], [-1, 768, 1, 75, 1, 1, 1], [-1, 768, 1, 80, 1, 1, 1], [-1, 768, 1, 85, 1, 1, 1], [-1, 768, 1, 90, 1, 1, 1], [-1, 768, 1, 95, 1, 1, 1], [-1, 768, 1, 100, 1, 1, 1]]
dna_default = [[-1, 320, 0, 3, 1, 1, 0], [-1, 320, 1, 3, 3, 1, 1], [-1, 384, 1, 6, 1, 1, 1], [-1, 448, 1, 9, 1, 1, 1], [-1, 512, 1, 12, 1, 1, 1], [-1, 576, 1, 15, 1, 1, 1], [-1, 640, 1, 18, 1, 1, 1], [-1, 704, 1, 21, 1, 1, 1], [-1, 768, 1, 24, 1, 1, 1], [-1, 832, 1, 27, 1, 1, 1], [-1, 896, 1, 30, 1, 1, 1], [-1, 960, 1, 33, 1, 1, 1]]

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.orig = d

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

class dataloader(Dataset):
    def __init__(self, recfile="/tmp/train.hdf5", seq_len=4096, index=False, elen=342):
        self.recfile = recfile
        self.seq_len = seq_len
        self.index = index
        h5 = h5py.File(self.recfile, "r")
        self.len = len(h5["events"])
        h5.close()
        self.elen = elen
        print("Dataloader total events:", self.len, "seqlen:", self.seq_len, "event len:", self.elen)

    def __getitem__(self, index):
        h5 = h5py.File(self.recfile, "r")
        event = h5["events"][index]
        event_len = self.elen
        label = h5["labels"][index]
        label_len = h5["labels_len"][index]
        h5.close()
        if not self.index:
            return (event, event_len, label, label_len)
        else:
            return (event, event_len, label, label_len, index)

    def __len__(self):
        return self.len

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x *( torch.tanh(torch.nn.functional.softplus(x)))

class squeeze_excite(torch.nn.Module):
    def __init__(self, in_channels = 512, size=1, reduction="/16", activation=torch.nn.GELU):
        super(squeeze_excite, self).__init__()
        self.in_channels = in_channels
        self.avg = torch.nn.AdaptiveAvgPool1d(1)
        if type(reduction) == str:
            self.reductionsize = self.in_channels // int(reduction[1:])
        else:
            self.reductionsize = reduction
        self.fc1 = nn.Linear(self.in_channels, self.reductionsize)
        self.activation = activation() # was nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.reductionsize, self.in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg(x)
        x = x.permute(0,2,1)
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return input * x.permute(0,2,1)


class convblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, seperable=True, expansion=True, batchnorm=True, dropout=0.1, activation=torch.nn.GELU, sqex=True, squeeze=32, sqex_activation=torch.nn.GELU, residual=True):
        # no bias?
        super(convblock, self).__init__()
        self.seperable = seperable
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        self.squeeze = squeeze
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.doexpansion = expansion
        # fix self.squeeze
        dwchannels = in_channels
        if seperable:
            if self.doexpansion and self.in_channels != self.out_channels:
                self.expansion = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False)
                self.expansion_norm = torch.nn.BatchNorm1d(out_channels)
                self.expansion_act = self.activation()
                dwchannels = out_channels 

            self.depthwise = torch.nn.Conv1d(dwchannels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=out_channels//groups)
            if self.batchnorm:
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            self.pointwise = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=dilation, bias=bias, padding=0)
            if self.batchnorm:
                self.bn2 = torch.nn.BatchNorm1d(out_channels)
            self.act2 = self.activation()
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        else:
            self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            if self.batchnorm:
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        if self.residual and self.stride == 1:
            self.rezero = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x):
        orig = x

        if self.seperable:
            if self.in_channels != self.out_channels and self.doexpansion:
                x = self.expansion(x)
                x = self.expansion_norm(x)
                x = self.expansion_act(x)
            x = self.depthwise(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.squeeze:
                x = self.sqex(x)
            x = self.pointwise(x)
            if self.batchnorm: x = self.bn2(x)
            x = self.act2(x) 
            if self.dropout: x = self.drop(x)
        else:
            x = self.conv(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.dropout: x = self.drop(x)

        if self.residual and self.stride == 1 and self.in_channels == self.out_channels and x.shape[2] == orig.shape[2]:
            return orig + self.rezero * x # rezero
            #return orig + x # normal residual
        else:
            return x

def activation_function(activation):
    if activation == "mish":
        return Mish
    elif activation == "swish":
        return Swish
    elif activation == "relu":
        return torch.nn.ReLU
    elif activation == "gelu":
        return torch.nn.GELU
    else:
        print("Unknown activation type:", activation)
        sys.exit(1)
    
class network(nn.Module):
    def __init__(self, config=defaultconfig, arch=rna_default, seqlen=4096, debug=False):
        super().__init__()
        if debug: print("Initializing network")
        
        self.seqlen = seqlen
        self.vocab = config.vocab
        
        self.bn = nn.BatchNorm1d

        # [P, Channels, Separable, kernel_size, stride, sqex, dropout]
        # P = -1 kernel_size//2, 0 none, >0 used as padding
        # Channels
        # seperable = 0 False, 1 True
        # kernel_size
        # stride
        # sqex = 0 False, 1 True
        # dropout = 0 False, 1 True
        if arch == None: arch = rna_default

        activation = activation_function(config.activation.lower())
        sqex_activation = activation_function(config.sqex_activation.lower())

        self.convlayers = nn.Sequential()
        in_channels = 1
        convsize = self.seqlen

        for i, layer in enumerate(arch):
            paddingarg = layer[0]
            out_channels = layer[1]
            seperable = layer[2] 
            kernel = layer[3]
            stride = layer[4]
            sqex = layer[5]
            dodropout = layer[6]
            expansion = True

            if dodropout: dropout = config.dropout
            else: dropout = 0
            if sqex: squeeze = config.sqex_reduction
            else: squeeze = 0

            if paddingarg == -1:
                padding = kernel // 2
            else: padding = paddingarg
            if i == 0: expansion = False

            convsize = (convsize + (padding*2) - (kernel-stride))//stride
            if debug:
                print("padding:", padding, "seperable:", seperable, "ch", out_channels, "k:", kernel, "s:", stride, "sqex:", sqex, "drop:", dropout, "expansion:", expansion)
                print("convsize:", convsize)
            self.convlayers.add_module("conv"+str(i), convblock(in_channels, out_channels, kernel, stride=stride, padding=padding, seperable=seperable, activation=activation, expansion=expansion, dropout=dropout, squeeze=squeeze, sqex_activation=sqex_activation, residual=True))
            in_channels = out_channels
            self.final_size = out_channels
         
        self.final = nn.Linear(self.final_size, len(self.vocab))
        if debug: print("Finished init network")

    def forward(self, x):
        #x = self.embedding(x)
        x = self.convlayers(x)
        x = x.permute(0,2,1)
        x = self.final(x)
        x = torch.nn.functional.log_softmax(x, 2)
        return x.permute(1, 0, 2)

counter = 0

def get_checkpoint(epoch, model, optimizer, scheduler):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }
    return checkpoint

def get_config(model, config):
    config = {
            "state_dict": model.state_dict(),
            "config": config
            }
    return config