from torch.utils.data import Dataset
import h5py
import numpy as np
import typing
from typing import Union

class MelchiorDataset(Dataset):
    def __init__(self, path:str="data/rna-train.hdf5", seq_len:int=4096, index:bool=False, event_len:int=342) -> None:
        self.path = path
        self.seq_len = seq_len
        self.index = index
        h5 = h5py.File(self.path, "r")
        self.len = len(h5["events"])
        h5.close()
        self.event_len = event_len
        print(f"Total events: {self.len}, Sequence length: {self.seq_len}, Event length: {self.event_len}")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: Union[None, int]) -> tuple[np.ndarray, int, np.ndarray, int, int]:
        h5 = h5py.File(self.path, "r")
        event = h5["events"][idx]
        event_len = self.event_len
        label = h5["labels"][idx]
        label_len = h5["labels_len"][idx]
        h5.close()
        if not self.index:
            return (event, event_len, label, label_len)
        else:
            return (event, event_len, label, label_len, idx)
