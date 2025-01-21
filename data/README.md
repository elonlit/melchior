# Data Directory Documentation

This directory contains all datasets and scripts necessary for training, validation, and testing of the Melchior basecaller.

### Training and Validation Data
To download the training and validation datasets:
```bash
source data/train_val/download_train_val.sh
```

### Test Data
To download the test FAST5 files:
```bash
source data/test/download_test.sh
```

## Dataset Components

### Sanity Check
- Located in `sanity_check/`
- Contains small test datasets for quick verification

### Test Dataset
- Located in `test/`
- Contains reference transcriptomes for multiple organisms:
  - Arabidopsis
  - Human
  - Mouse
  - Poplar
  - Yeast
- Each transcriptome includes both FASTA and index files

### Training and Validation
- Located in `train_val/`
- Contains the main datasets used for model training and validation

> [!IMPORTANT]
> Download all required datasets before running training or evaluation scripts.

> [!NOTE]
> The reference transcriptomes are automatically indexed during download.