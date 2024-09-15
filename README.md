# Melchior

A hybrid Mamba-Transformer model for RNA base calling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![arXiv](https://img.shields.io/badge/arXiv-2023.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2023.xxxxx)

## Features

- High-accuracy RNA base calling
- Efficient state space modeling
- Scalable to large datasets
- Easy-to-use interface

## Quick Start

```bash
source setup.sh # Set up the new environment
source clean.sh # Clean the existing environment
```

### Datasets

```bash
source data/download_train_val.sh
```

This script downloads the training and validation splits into the `data/` directory. Note that `aria2c` is used to improve the download speed, which may not be available on all systems:

```bash
sudo apt-get install aria2
```

In preparation to run the evaluation script, download the test fast5 files as well:

```bash
source data/download_test.sh
```

## Usage

```python
from melchior import RNABaseCaller

base_caller = RNABaseCaller()
signal_data = load_signal_data("path/to/signal_file")
sequence = base_caller.call_bases(signal_data)
print(f"Predicted RNA sequence: {sequence}")
```

## Installation

```bash
pip install melchior
```

For detailed installation instructions, see our [Installation Guide](docs/installation.md).

## Training

To train the model from scratch, run:

```bash
python -m utils.train

Training utility for Melchior

options:
  -h, --help            show this help message and exit
  --model {melchior,rodan}
  --state_dict STATE_DICT
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --save_path SAVE_PATH
```

If you have access to a Slurm cluster, run:

```bash
./train.sh
```

## Reproducible Evaluation

Weights for the 28 million parameter model are available on HuggingFace.

Download the test set fasta and fast5 files in /data/test/:

```bash
./download_test.sh
```

Minimap2 may be necessary, which is not available on all systems:

```bash
sudo apt install minimap2
```

Then run the evaluation script to basecall the .fast5 files, align the basecalled sequences to the reference transcriptomes, and calculate the accuracy:

```bash
eval/run_tests.sh
```

Note: the evaluation script must be run from the root directory of the project.

## Contributing

We welcome contributions. Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

Melchior is released under the [MIT License](LICENSE).

## Citation

If you use Melchior in your research, please cite:

```bibtex
@article{melchior2024,
  title={Melchior: A State Space Model for RNA Base Calling},
  author={Litman, Elon},
  journal={arXiv preprint arXiv:2023.xxxxx},
  year={2024}
}
```