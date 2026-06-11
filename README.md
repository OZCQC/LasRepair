# DataRepair

A multi-model approach for data repair using sequence-to-sequence learning and MICE (Multivariate Imputation by Chained Equations).

## Overview

This project implements an iterative data repair system that uses T5-based models to fix errors in tabular data. The system employs a multi-model approach where each column is handled by a separate model, allowing for specialized repair strategies for different data types.

## Features

- Multi-model approach for data repair
- Custom T5 model implementation with processing layers
- Support for multiple datasets (flights, beers, hospital, rayyan)
- Iterative repair process
- GPU acceleration support
- LoRA fine-tuning for efficient model adaptation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OZCQC/datarepair.git
cd datarepair
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script can be run with different datasets:

```bash
python multi.py [dataset_name] [gpu_device]
```

Available datasets:
- flight
- beers
- hospital
- rayyan

Example:
```bash
python multi.py flight cuda:0
```

## Project Structure

- `multi.py`: Main implementation of the multi-model repair system
- `custom_model.py`: Custom T5 model implementation
- `dataset.py`: Dataset handling and preprocessing
- `utils.py`: Utility functions
- `dataset/`: Directory containing the datasets
- `models_ckp/`: Directory for model checkpoints
- `result/`: Directory for repair results
- `logs/`: Directory for log files

## Requirements

See `requirements.txt` for the full list of dependencies. Key requirements include:
- PyTorch
- Transformers
- PEFT
- pandas
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

