import h5py
import numpy as np

with h5py.File('data/rna-train.hdf5', 'r') as f:
    for key in ['events', 'labels', 'labels_len']:
        dataset = f[key]
        print(f"\nDataset: {key}")
        print(f"Shape: {dataset.shape}")
        print(f"Dtype: {dataset.dtype}")
        
        # Print a sample of the data
        if len(dataset.shape) == 1:
            print("First few elements:")
            print(dataset[:5])
        elif len(dataset.shape) == 2:
            print("First few rows:")
            print(dataset[:5])
        else:
            print("First item shape:", dataset[0].shape)
            print("First item first few elements:")
            print(dataset[0].flatten()[:5])