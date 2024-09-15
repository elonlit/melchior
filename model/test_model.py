import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, ctc_loss
import numpy as np

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
import h5py
from torch.autograd import Variable
import sys, argparse, shutil, pickle, os, copy, math
from collections import OrderedDict

config = {"name":"default", "seqlen":4096, "epochs":30, "optimizer":"ranger", "lr":2e-3, "weightdecay":0.01, "batchsize":30, "dropout": 0.1, "activation":"mish", "sqex_activation":"mish", "sqex_reduction":32, "trainfile":"rna-train.hdf5", "validfile":"rna-valid.hdf5", "amp":False, "scheduler":"reducelronplateau", "scheduler_patience":1, "scheduler_factor":0.5, "scheduler_threshold":0.1, "scheduler_minlr": 1e-05, "scheduler_reduce":2, "gradclip":0, "train_loopcount": 1000000, "valid_loopcount": 1000, "tensorboard":False, "saveinit":False,
        "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}

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
    def __init__(self, config=None, arch=None, seqlen=4096, debug=False):
        super().__init__()
        if debug: print("Initializing network")
        
        self.seqlen = seqlen
        self.vocab = config['vocab']
        
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

        activation = activation_function(config['activation'])
        sqex_activation = activation_function(config['sqex_activation'])

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

            if dodropout: dropout = config['dropout']
            else: dropout = 0
            if sqex: squeeze = config['sqex_reduction']
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


if __name__ == '__main__':
    model = network(config=config)
    x = torch.randn(32, 1, 4096)
    y = model(x)
    print(y.shape)

    model = Model()
    x = torch.randn(32, 1, 4096)
    y = model(x)
    print(y.shape)
