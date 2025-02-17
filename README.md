# Melchior

A hybrid Mamba-Transformer model for RNA base calling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.01.11.632456-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.01.11.632456v1)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-Melchior-blue)](https://huggingface.co/elonlit/Melchior)

## Features

- High-accuracy RNA base calling
- Efficient state space modeling
- Open weights, checkpoints, logs
- 100% reproducible

## Quick Start

```bash
source setup.sh # Set up the new environment
source clean.sh # Clean the existing environment
```

### Datasets

```bash
source data/train_val/download_train_val.sh
```

This script downloads the training and validation splits into the `data/train_val/` directory.

> [!NOTE]
> `aria2c` is used to improve the download speed, which may not be available on all systems: ```sudo apt-get install aria2```

In preparation to run the evaluation script, download the test fast5 files as well:

```bash
source data/test/download_test.sh
```

## Training

To train the model from scratch, run:

```bash
python -m utils.train [OPTIONS]
```

### Options:

- `--model`: Choose model architecture (`melchior` or `rodan`). Default: `melchior`
- `--state_dict`: Path to initial state dict. Default: None
- `--epochs`: Number of training epochs. Default: 10
- `--batch_size`: Batch size for training. Default: 16
- `--lr`: Learning rate. Default: 0.001
- `--weight_decay`: Weight decay for optimizer. Default: 0.1
- `--save_path`: Path to save model checkpoints. Default: `models/{model}`

### Example:

```bash
python -m utils.train --model melchior --epochs 10 --batch_size 32 --lr 0.002
```

## Reproducible Evaluation

Weights for the 134 million parameter model are available on HuggingFace.

Download the test set fasta and fast5 files in ```./data/test/```:

```bash
./download_test.sh
```

> [!NOTE]  
> `minimap2` is necessary, which may not available on all systems: ```sudo apt install minimap2```

Then run the evaluation script to basecall the .fast5 files, align the basecalled sequences to the reference transcriptomes, and calculate the accuracy:

```bash
eval/run_tests.sh
```

> [!IMPORTANT]  
> To benchmark against ONT's proprietary basecallers, you need to download them first:
> 
> For Guppy:
> ```
> ./basecallers/download_guppy.sh [-g|-c]
>   -g: Download GPU version
>   -c: Download CPU version
> ```
> 
> Run these scripts before proceeding with the evaluation.

## Contributing

We welcome contributions. Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

Melchior is released under the [MIT License](LICENSE).

## Citation

If you use Melchior in your research, please cite:

```bibtex
@article{melchior2025,
  title={Melchior: A Hybrid Mamba-Transformer RNA Basecaller},
  author={Litman, Elon},
  journal={bioRxiv preprint bioRxiv:2025.01.11.632456},
  year={2025}
}
```