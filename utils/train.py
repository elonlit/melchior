import os
import numpy as np
import torch
from model.melchior import Melchior, MelchiorModule
from torchinfo import summary
from model.rodan import RodanModule
from utils import MelchiorDataset
from torch.utils.data import DataLoader
from typing import Union
from torch.nn.functional import ctc_loss
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from lightning.pytorch.loggers import WandbLogger
import argparse

# Saves the model checkpoint
def get_checkpoint(epoch, model, optimizer, scheduler):
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

def train_melchior(state_dict:Union[None, str] = None, 
          epochs:int=20,
          batch_size:int=12,
          lr:float=2e-3,
          weight_decay:float=0.01,
          save_path:Union[None, str]="models/melchior",
          num_gpus:Union[None, int]=None) -> tuple[list, list, list]:
    
    # Create data module
    data_train = MelchiorDataset("data/train_val/rna-train.hdf5")
    data_valid = MelchiorDataset("data/train_val/rna-valid.hdf5")
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=16)

    # Create model
    model = MelchiorModule(lr=lr, weight_decay=weight_decay, train_loader=train_loader, epochs=epochs, accumulate_grad_batches=args.accumulate_grad_batches)

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Create trainer
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename='{epoch}-{val_loss:.2f}', save_top_k=-1, monitor='val_loss')
    wandb_logger = WandbLogger(log_model="all")
    
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=epochs,
        accelerator='gpu',
        devices=num_gpus,
        strategy='ddp',
        callbacks=[checkpoint_callback, swa_callback],
        log_every_n_steps=1000,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
    )

    wandb.finish()
    wandb.init(project="melchior")
    wandb.watch(model, log='all')

    # Train the model
    if state_dict:
        print("Resuming training from checkpoint...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=state_dict)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training complete.")
    return trainer.callback_metrics['train_loss'], trainer.callback_metrics['val_loss'], model.lr_schedulers().get_last_lr()

if __name__ == '__main__':
    if not os.path.isfile('README.md'):
        print("Please run this script from the root directory of the project.")
        exit(1)

    parser = argparse.ArgumentParser(description="Training utility for Melchior")
    parser.add_argument("--model", type=str, default="melchior", choices=["melchior"]) # Add argument for model {rodan, melchior}
    parser.add_argument("--state_dict", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="models/melchior")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available GPUs)")
    args = parser.parse_args()

    if args.model == "melchior":
        train_melchior(args.state_dict, args.epochs, args.batch_size, args.lr, args.weight_decay, "models/melchior", args.num_gpus)
    else:
        print("Invalid model specified.")
        exit(1)