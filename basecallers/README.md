# 🧬 Basecallers Directory Guide

This directory contains various models for basecalling ONT sequencing data.

## 📁 Directory Structure
```
basecallers/
├── README.md           # Documentation file
├── download_basecallers.sh  # Master script for downloading all basecallers
├── download_gcrtcall.sh     # Script for GCRTcall download
├── download_guppy.sh        # Script for Guppy download
├── download_melchior.sh     # Script for Melchior download
├── GCRTcall/               # GCRTcall model directory
│   └── GCRTcall_ckpt.pt    # GCRTcall model weights
└── rodan/                  # RODAN model directory
    └── rna.torch           # RODAN model weights
```

## 📥 Installation
Each basecaller has its own download script for easy installation:
```bash
# Download all basecallers
./download_basecallers.sh

# Or download individual basecallers
./download_gcrtcall.sh
./download_melchior.sh
./download_guppy.sh
```

> [!IMPORTANT]  
> The download script for Guppy will, by default, download the GPU version.
> ```
> ./basecallers/download_guppy.sh [-g|-c]
>   -g: Download GPU version
>   -c: Download CPU version
> ```
> To change this, adjust the command options.

---
*Note: Make sure you have sufficient storage space and required dependencies before downloading the models.*