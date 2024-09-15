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

# """
# Validates the model on the validation set and returns the loss.
# """
# def validate(model:torch.nn.Module, device:torch.device, batch_size:int, epoch:int) -> float:
#     print("Running validation...")
#     model.eval()
#     data = MelchiorDataset("data/rna-valid.hdf5")
#     val_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     total_loss = 0
#     loop_count = 0

#     with torch.no_grad():
#         for i, (event, event_len, label, label_len) in enumerate(val_loader):
#             event = torch.unsqueeze(event, 1)
#             label = label[:, :max(label_len)]

#             event, label, event_len, label_len = event.to(device), label.to(device), event_len.to(device), label_len.to(device)

#             output = model(event)

#             loss = ctc_loss(output, label, event_len, label_len, reduction="mean", blank=0, zero_infinity=True)

#             total_loss += loss.item()
#             loop_count += 1

#     print(f"Epoch {epoch} validation loss: {total_loss / loop_count}")
#     return np.float(total_loss / loop_count)



# # Trains the model
# def train(state_dict:Union[None, dict] = None, 
#           epochs=10, 
#           batch_size=32,
#           lr=1e-3,
#           weight_decay=1e-5,
#           save_path="models/melchior.pth") -> tuple[list, list, list]:
    
#     train_loss = []
#     val_loss = []
#     learning_rates = []

#     device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#     print("Using device:", device)

#     # model = DataParallel(Melchior().to(device))
#     model = Melchior(in_chans=1, embed_dim=512, depth=12)

#     print("Model:",model)
#     print("Model parameters:", sum(p.numel() for p in model.parameters()))

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     data = MelchiorDataset()
#     train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

#     if state_dict:
#         print("Loading state dict...")
#         checkpoint = torch.load(state_dict)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         loop_count = 0

#         learning_rates.append(optimizer.param_groups[0]['lr'])

#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

#         for i, (event, event_len, label, label_len) in enumerate(progress_bar):
#             event = torch.unsqueeze(event, 1)
#             label = label[:, :max(label_len)]

#             event, label, event_len, label_len = event.to(device), label.to(device), event_len.to(device), label_len.to(device)

#             optimizer.zero_grad()

#             output = model(event)

#             loss = ctc_loss(output, label, event_len, label_len, reduction="mean", blank=0, zero_infinity=True)

#             total_loss += loss.item()
#             loss.backward()

#             optimizer.step()
#             loop_count += 1

#             # Update progress bar description
#             progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

#         val = validate(model, device, batch_size, epoch)
#         val_loss.append(val)

#         scheduler.step(val)
#         print(f"Epoch {epoch+1}/{epochs} training loss: {total_loss / loop_count:.4f}")
#         train_loss.append(total_loss / loop_count)

#         print(f"Saving epoch {epoch+1} checkpoint...")
#         torch.save(get_checkpoint(epoch, model, optimizer, scheduler), save_path)

#     print("Training complete.")
#     return train_loss, val_loss, learning_rates

# if __name__ == '__main__':
#     train(None, 10, 32, 1e-3, 1e-5, "models/melchior.pth")

def train_melchior(state_dict:Union[None, str] = None, 
          epochs=5, 
          batch_size=16,
          lr=1e-4,
          weight_decay=1e-5,
          save_path="models/melchior"):
    
    # Create data module
    data_train = MelchiorDataset("data/train_val/rna-train.hdf5")
    data_valid = MelchiorDataset("data/train_val/rna-valid.hdf5")
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=4)  # Assuming you have a validation set

    # Create model
    model = MelchiorModule(lr=lr, weight_decay=weight_decay, train_loader=train_loader, epochs=epochs)

    # Load state dict if provided
    if state_dict:
        print("Loading state dict...")
        model = MelchiorModule.load_from_checkpoint(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Create trainer
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename='{epoch}-{val_loss:.2f}', save_top_k=3, monitor='val_loss')
    wandb_logger = WandbLogger(log_model="all")
    
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=epochs,
        accelerator='gpu',
        devices=2, # Use 2 GPUs
        strategy='ddp',
        callbacks=[checkpoint_callback],
        log_every_n_steps=1000,
        accumulate_grad_batches=4,
        precision=16,
        logger=wandb_logger,
    )

    wandb.init(project="melchior")
    wandb.watch(model, log='all')

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training complete.")
    return trainer.callback_metrics['train_loss'], trainer.callback_metrics['val_loss'], model.lr_schedulers().get_last_lr()

def train_rodan(state_dict:Union[None, str] = None, 
          epochs=5, 
          batch_size=16,
          lr=1e-4,
          weight_decay=1e-5,
          save_path="models/rodan"):
    
    # Create data module
    data_train = MelchiorDataset("data/train_val/rna-train.hdf5")
    data_valid = MelchiorDataset("data/train_val/rna-valid.hdf5")

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = RodanModule(lr=lr, weight_decay=weight_decay)

    # Load state dict if provided
    if state_dict:
        model = RodanModule.load_from_checkpoint(state_dict)

    # Create trainer
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename='{epoch}-{val_loss:.2f}', save_top_k=3, monitor='val_loss')
    wandb_logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        profiler="simple",
        max_epochs=epochs,
        accelerator='gpu',
        devices=2, # Use 2 GPUs
        callbacks=[checkpoint_callback],
        log_every_n_steps=1000,
        accumulate_grad_batches=4,
        precision=16,
        logger=wandb_logger,
    )

    wandb.init(project="rodan")
    wandb.watch(model, log='all')

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training complete.")
    return trainer.callback_metrics['train_loss'], trainer.callback_metrics['val_loss'], model.lr_schedulers().get_last_lr()

    
if __name__ == '__main__':
    if not os.path.isfile('README.md'):
        print("Please run this script from the root directory of the project.")
        exit(1)

    parser = argparse.ArgumentParser(description="Training utility for Melchior")
    parser.add_argument("--model", type=str, default="melchior", choices=["melchior", "rodan"]) # Add argument for model {rodan, melchior}
    parser.add_argument("--state_dict", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="models/melchior")
    args = parser.parse_args()

    if args.model == "melchior":
        train_melchior(args.state_dict, args.epochs, args.batch_size, args.lr, args.weight_decay, "models/melchior")
    elif args.model == "rodan":
        train_rodan(args.state_dict, args.epochs, args.batch_size, args.lr, args.weight_decay, "models/rodan")
    else:
        print("Invalid model specified.")
        exit(1)