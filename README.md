SAMBA: A Graph-Mamba Approach for Stock Price Prediction  
[**Paper**](https://arxiv.org/pdf/2410.03707) | [**Dataset**](https://www.kaggle.com/datasets/ehoseinz/stock-market-prediction/data)
===



<p align="center">
  <img src="abc.PNG" alt="Title of the Picture">
  <br>
</p>

## 📖 About

This repository contains the modular implementation of **SAMBA**, a novel architecture that combines State-space Mamba models with Graph Neural Networks for stock price prediction. This work is based on the paper **"Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction"** accepted for publication in _IEEE ICASSP 2025_.

🔗 **Original Paper Repository**: [https://github.com/Ali-Meh619/SAMBA](https://github.com/Ali-Meh619/SAMBA)

## 🎯 Overview

SAMBA (State-space Mamba with Graph Neural Networks) is designed for stock price prediction using real-world financial market data. The model leverages:

- 🧠 **Mamba blocks** for efficient sequence modeling with selective state spaces
- 🕸️ **Graph Neural Networks** with Chebyshev polynomials for spatial relationships
- 📊 **Gaussian kernel-based adjacency matrices** for adaptive graph learning
- ⚡ **Bidirectional processing** for enhanced temporal understanding

## 🏗️ Architecture

The model consists of several key components:

1. 🧠 **Mamba Backbone**: Processes temporal sequences using selective state space models
2. 🕸️ **Graph Convolution Layers**: Capture spatial dependencies using Chebyshev polynomials
3. 📊 **Adaptive Adjacency Matrix**: Learns graph structure using Gaussian kernels
4. 🔗 **Residual Connections**: Enable deep network training with skip connections

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd samba-stock-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure you have a CUDA-compatible GPU for optimal performance.** 🎮

## 💻 Usage

### ⚡ Quick Start

1. **📦 Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **📁 Create Dataset folder and add your CSV files:**
   ```bash
   mkdir Dataset
   # Keep the SAMBA feature builder and copy your CSV files to the Dataset folder:
   # - samba_feature_builder.py
   # - combined_dataframe_IXIC.csv
   # - combined_dataframe_NYSE.csv  
   # - combined_dataframe_DJI.csv
   ```

3. **🚀 Run the model:**
   ```bash
   python main.py
   ```

### 🎯 Basic Training

**Run the model:**
```python
from main import main

# Run training with paper configuration
main()
```

### ⚙️ Custom Configuration

You can modify the training configuration in `main.py` or `paper_config.py`:

```python
# In main.py, the configuration is loaded from paper_config.py
model_args, config = get_paper_config()

# You can modify the config before training
config.epochs = 500  # Reduce epochs for faster training
config.batch_size = 64  # Increase batch size
```

### 📊 Data Format

> **⚠️ Important**: Create a `Dataset` folder and place your CSV files in it.

The model expects SAMBA combined CSV data with the following format:
- 📅 **Date column** as index
- 🏷️ **Name column** (will be removed during preprocessing)
- 💰 **Price column** for target values
- 📈 **82 feature columns** containing target-market technical features and shared market variables

**Example:**
```csv
Date,Price,Vol.,weekday,mom,mom1,...,DE6
2010-01-04,2308.42,-0.14709,0,-0.000126,...,1.23
2010-01-05,2308.71,0.034934,1,0.003311,...,1.12
```

> **💡 Note**: The `num_nodes` parameter is automatically determined from the input data shape (number of features), so you don't need to specify it manually.

### SAMBA Feature Builder

`Dataset/samba_feature_builder.py` builds SAMBA-compatible combined feature files when the required raw source series are available. It computes the target-market technical features, merges shared external variables, and derives the SAMBA term-spread and default-spread features.

Use `legacy` mode to match the published SAMBA dataset convention:

```bash
python Dataset/samba_feature_builder.py \
  --target-csv raw_ixic.csv \
  --target-name IXIC \
  --external-csv raw_external_features.csv \
  --external-mode raw \
  --external-alignment reverse-position \
  --mode legacy \
  --output Dataset/combined_dataframe_IXIC.csv
```

Use `causal` mode for chronological feature engineering on new datasets:

```bash
python Dataset/samba_feature_builder.py \
  --target-csv raw_ixic.csv \
  --target-name IXIC \
  --external-csv raw_external_features.csv \
  --external-mode raw \
  --external-alignment date \
  --mode causal \
  --output Dataset/combined_dataframe_IXIC.csv
```

Exact reproduction of the published combined CSVs requires the same raw source series, vendor calendars, missing values, historical snapshots, and rounding used to create those files.

### 📈 Available Datasets

This repository is configured to work with three real-world datasets from the US stock market with **82 daily stock features**:

**📁 Folder Structure:**
```
Dataset/
├── samba_feature_builder.py       # SAMBA feature construction utility
├── combined_dataframe_IXIC.csv    # 📊 NASDAQ Composite Index
├── combined_dataframe_NYSE.csv    # 🏛️ New York Stock Exchange
└── combined_dataframe_DJI.csv     # 📈 Dow Jones Industrial Average
```

**📅 Dataset Period**: January 2010 to November 2023  
**🔢 Features**: 82 daily stock features including technical indicators, market data, and financial metrics

Each dataset contains comprehensive historical price data with multiple technical indicators as features, providing rich information for the Graph-Mamba model to learn complex market patterns.

## 🧩 Model Components

### 🔧 Core Modules

- 📋 `config/`: Configuration classes for model and training parameters
- 🧠 `models/`: Model implementations (SAMBA, Mamba, Graph layers)
- 🛠️ `utils/`: Utility functions (data loading, metrics, logging)
- 🏃 `trainer/`: Training loop and evaluation

### 🎯 Key Classes

- 🚀 `SAMBA`: Main model combining Mamba and GNN
- 🧠 `Mamba`: State-space sequence model
- 🔗 `MambaBlock`: Individual Mamba block with selective scanning
- 🕸️ `gconv`: Graph convolution with Chebyshev polynomials
- 🏃 `Trainer`: Training and evaluation pipeline

## 📊 Metrics

The model evaluates performance using:

- 📏 **MAE**: Mean Absolute Error
- 📐 **RMSE**: Root Mean Squared Error
- 🔗 **IC**: Information Coefficient (Pearson correlation)
- 📈 **RIC**: Rank Information Coefficient (Spearman correlation)

## ⚙️ Configuration

### 🧠 Model Parameters

- 🔢 `d_model`: Model dimension
- 📚 `n_layer`: Number of Mamba layers
- 🎯 `vocab_size`: Number of features (automatically determined from input data)
- 📥 `seq_in`: Input sequence length
- 📤 `seq_out`: Output sequence length
- 🏠 `d_state`: State dimension
- 📈 `expand`: Expansion factor
- 🧮 `cheb_k`: Chebyshev polynomial order

### 🏃 Training Parameters

- 🔄 `epochs`: Number of training epochs
- 📚 `lr_init`: Initial learning rate
- 📦 `batch_size`: Training batch size
- ⏹️ `early_stop`: Enable early stopping
- ⏰ `early_stop_patience`: Early stopping patience

## 📈 Results

The model outputs results to:
- 📄 `samba_results.txt`: Performance metrics
- 💾 `./best_model.pth`: Best model checkpoint
- 📺 Console logs: Training progress and final metrics

## 📁 File Structure

```
├── 📂 Dataset/                # SAMBA datasets and feature builder
│   ├── samba_feature_builder.py
│   ├── 📊 combined_dataframe_IXIC.csv
│   ├── 🏛️ combined_dataframe_NYSE.csv
│   └── 📈 combined_dataframe_DJI.csv
├── 📂 config/
│   ├── __init__.py
│   └── model_config.py
├── 📂 models/
│   ├── __init__.py
│   ├── 🚀 samba.py
│   ├── 🧠 mamba.py
│   ├── 🔗 mamba_block.py
│   ├── 🕸️ graph_layers.py
│   └── 📏 normalization.py
├── 📂 utils/
│   ├── __init__.py
│   ├── 📊 data_utils.py
│   ├── 📈 metrics.py
│   ├── 📝 logger.py
│   └── 🛠️ model_utils.py
├── 📂 trainer/
│   ├── __init__.py
│   └── 🏃 trainer.py
├── 🚀 main.py                 # Main execution file
├── 📋 paper_config.py         # Paper-specific configuration
├── 🧪 test_system.py          # System test
├── 📦 requirements.txt
└── 📖 README.md
```

## 📦 Dependencies

- 🔥 **PyTorch** >= 1.9.0
- 🔢 **NumPy** >= 1.21.0
- 🐼 **Pandas** >= 1.3.0
- 📊 **Matplotlib** >= 3.4.0
- 🧮 **einops** >= 0.4.0
- 💾 **h5py** >= 3.1.0

## 📚 Citation

If you find our paper and code useful, please kindly cite our paper as follows:

```bibtex
@inproceedings{SAMBA,
  title={Mamba Meets Financial Markets: {A} {G}raph-{M}amba Approach for Stock Price Prediction},
  author={Mehrabian, Ali and Hoseinzade, Ehsan and Mazloum, Mahdi and Chen, Xiaohong},
  booktitle={Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
address={Hyderabad, India},
  month={Apr.},
  year={2025}
}
```

**📄 Paper**: "Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction"  
**🏛️ Conference**: IEEE ICASSP 2025  
**👥 Authors**: Ali Mehrabian, Ehsan Hoseinzade, Mahdi Mazloum, Xiaohong Chen


## 📞 Contact

Please feel free to contact us if you have any questions:

- 👨‍💻 **Ali Mehrabian**: alimehrabian619@yahoo.com, ali.mehrabian@vectorinstitute.ai
- 🔗 **Original Repository**: [https://github.com/Ali-Meh619/SAMBA](https://github.com/Ali-Meh619/SAMBA)

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. 🧪 Add tests if applicable
5. 📤 Submit a pull request

## 🐛 Issues

If you encounter any issues, please:
1. 🔍 Check the existing issues
2. 📝 Create a new issue with detailed description
3. 💻 Include system information and error logs

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for the financial AI community

</div>
