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
import subprocess
import re
torch.set_float32_matmul_precision('medium')

# Saves the model checkpoint
def get_checkpoint(epoch, model, optimizer, scheduler):
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

class SanityCheckCallback(pl.Callback):
    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.output_file = "eval/sanity_check_outputs/accuracy.txt"

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):        
        self.run_sanity_check(pl_module)

    def run_sanity_check(self, pl_module):
        if os.path.exists(self.script_path):
            try:
                print(f"Running sanity check script: {self.script_path}")
                subprocess.run([self.script_path, "melchior"], 
                               check=True)  # Set a timeout
                
                # Read the output from the file
                with open(self.output_file, 'r') as f:
                    output = f.read()
                                
                metrics = self.parse_sanity_check_output(output)
                
                if not metrics:
                    print("Warning: No metrics were parsed from the output.")
                
                wandb_metrics = {}
                for key, value in metrics.items():
                    metric_name = f'sanity_check_{key}'
                    pl_module.log(metric_name, value, on_epoch=True, sync_dist=True)
                    wandb_metrics[metric_name] = value
                
                # Log directly to wandb
                if wandb.run is not None:
                    wandb.log(wandb_metrics)
                else:
                    print("Warning: wandb.run is None. Make sure wandb is properly initialized.")
                
                print("Sanity check completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Sanity check failed with error: {e}")
                print(f"Return code: {e.returncode}")
            except subprocess.TimeoutExpired as e:
                print("Sanity check timed out")
            except Exception as e:
                print(f"Unexpected error during sanity check: {str(e)}")
        else:
            print(f"Sanity check script not found at {self.script_path}")

    def parse_sanity_check_output(self, output: str) -> dict:
        metrics = {}
        
        # Parse total and accuracies
        total_accuracy_match = re.search(r"Total: (\d+) Median accuracy: ([\d.]+) Average accuracy: ([\d.]+) std: ([\d.]+)", output)
        if total_accuracy_match:
            metrics['total'] = int(total_accuracy_match.group(1))
            metrics['median_accuracy'] = float(total_accuracy_match.group(2))
            metrics['average_accuracy'] = float(total_accuracy_match.group(3))
            metrics['std'] = float(total_accuracy_match.group(4))
        
        # Parse median errors
        median_errors_match = re.search(r"Median  - Mismatch: ([\d.]+) Deletions: ([\d.]+) Insertions: ([\d.]+)", output)
        if median_errors_match:
            metrics['median_mismatch'] = float(median_errors_match.group(1))
            metrics['median_deletions'] = float(median_errors_match.group(2))
            metrics['median_insertions'] = float(median_errors_match.group(3))
        
        # Parse average errors
        average_errors_match = re.search(r"Average - Mismatch: ([\d.]+) Deletions: ([\d.]+) Insertions: ([\d.]+)", output)
        if average_errors_match:
            metrics['average_mismatch'] = float(average_errors_match.group(1))
            metrics['average_deletions'] = float(average_errors_match.group(2))
            metrics['average_insertions'] = float(average_errors_match.group(3))
                
        if not metrics:
            print("Warning: No metrics were parsed from the output.")
            print("Please check if the sanity check script output matches the expected format.")
        
        return metrics

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
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Create model
    model = MelchiorModule(lr=lr, weight_decay=weight_decay, train_loader=train_loader, epochs=epochs, accumulate_grad_batches=args.accumulate_grad_batches)

    # Print model summary
    print(summary(model, input_size=(batch_size, 1, 4096)))

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Create trainer
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename='{epoch}-{val_loss:.2f}', save_top_k=-1, monitor='val_loss')
    wandb_logger = WandbLogger(log_model="all", entity="julian-q")
    
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    sanity_check_script_path = os.path.join(os.getcwd(), "eval", "sanity_check.sh")
    sanity_check_callback = SanityCheckCallback(sanity_check_script_path)

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=epochs,
        accelerator='gpu',
        devices=num_gpus,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, swa_callback, sanity_check_callback],
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="models/melchior")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available GPUs)")
    args = parser.parse_args()

    if args.model == "melchior":
        train_melchior(args.state_dict, args.epochs, args.batch_size, args.lr, args.weight_decay, "models/melchior", args.num_gpus)
    else:
        print("Invalid model specified.")
        exit(1)
