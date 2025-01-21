# 🧬 Basecallers Directory Guide

This directory contains various models for basecalling ONT sequencing data.

## 📁 Directory Structure
```
basecallers/
├── BASECALLERS.md           # Documentation file
├── download_basecallers.sh  # Master script for downloading all basecallers
├── download_gcrtcall.sh     # Script for GCRTcall download
├── download_guppy.sh        # Script for Guppy download
├── download_melchior.sh     # Script for Melchior download
├── GCRTcall/               # GCRTcall model directory
│   └── GCRTcall_ckpt.pt    # GCRTcall model weights
└── rodan/                  # Rodan model directory
    └── rna.torch           # RNA model weights
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

---
*Note: Make sure you have sufficient storage space and required dependencies before downloading the models.*