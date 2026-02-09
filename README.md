# Impute On-Demand: Adaptive Correlated Time Series Imputation for Changing Environments

Welcome to the repository for the paper **"Impute On-Demand: Adaptive Correlated Time Series Imputation for Changing Environments"**. This repository provides the official implementation, including code, environment setup, and dataset handling instructions.

---

## 🚀 Environment Setup

We strongly recommend using **Anaconda** or **Miniconda** to manage the project environment.

### Option 1: Quick Setup (Recommended)
This will create a conda environment named `adactsi_env` with all necessary dependencies, including GPU support (if available).

```bash
conda env create -f environment.yml
conda activate adactsi_env
```

### Option 2: Offline Setup (Pre-packed Environment)
If you have the pre-packed environment file (`packed_environment.tar.gz`), you can set it up without installing Conda or downloading dependencies.

1.  **Extract the environment**:
    ```bash
    mkdir -p adactsi_env
    tar -xzf packed_environment.tar.gz -C adactsi_env
    ```

2.  **Activate the environment**:
    ```bash
    source adactsi_env/bin/activate
    ```

3.  **Verify installation**:
    ```bash
    python --version
    ```



---

## 📂 Datasets

Please download the datasets from [here](https://drive.google.com/file/d/1zv6-qV_JIP2O2fFxHXfDHJ7Ee6vLWNcE/view?usp=sharing) and place them in the `datasets/` directory. Ensure the directory structure remains consistent with the current setup.

The project supports the following datasets:

| Dataset | Description | Config File |
| :--- | :--- | :--- |
| **PeMS-BA (PEMS04)** | Bay Area Traffic Flow | `config/adactsi/pems.yaml` |
| **PeMS-LA (PEMS07)** | Los Angeles Traffic Flow | `config/adactsi/pems.yaml` |
| **PeMS-SD (PEMS11)** | San Diego Traffic Flow | `config/adactsi/pems.yaml` |
| **Beijing Air Quality** | Air Quality Monitoring | `config/adactsi/beijing12.yaml` |
| **Vessel AIS** | Vessel Trajectory Data | `config/adactsi/vessel.yaml` |

---

## 🏃‍♂️ Running Experiments

Run experiments using the `run_adactsi.py` script. Below are the standard commands for each dataset.

### 1. PeMS Datasets (Traffic Flow)
```bash
# PeMS04
python run_adactsi.py --dataset-name pems_point --subdataset-name PEMS04 --config config/adactsi/pems.yaml

# PeMS07
python run_adactsi.py --dataset-name pems_point --subdataset-name PEMS07 --config config/adactsi/pems.yaml

# PeMS11
python run_adactsi.py --dataset-name pems_point --subdataset-name PEMS11 --config config/adactsi/pems.yaml
```

### 2. Beijing Air Quality
```bash
python run_adactsi.py --dataset-name beijing12 --config config/adactsi/beijing12.yaml
```

### 3. Vessel AIS Data
**Note**: To prevent memory overflow on standard machines, this dataset runs with `limit_nodes=20` by default. You can override this by adding `--limit-nodes <N>`.

```bash
python run_adactsi.py --dataset-name vessel --config config/adactsi/vessel.yaml
```


---

## 🖥️ GPU Configuration

You can control which device to use for training/inference via the `--gpu` flag.

| Value | Description | Example |
| :--- | :--- | :--- |
| **`0`** (Default) | Use GPU 0. Can specify multiple IDs for multi-GPU. | `--gpu 0` or `--gpu 0,1` |
| **`-1` / `cpu` / `none`** | Force CPU usage. | `--gpu cpu` |
| **`auto`** | Use GPU if available, otherwise fall back to CPU. | `--gpu auto` |

**Example**: Run on GPU 1:
```bash
python run_adactsi.py --dataset-name pems_point --subdataset-name PEMS04 --gpu 1
```

---

## ⚙️ Advanced Usage: Inference Complexity

AdaCTSi supports **Adaptive Inference**, allowing you to control the computational cost at inference time. You can specify one or more complexity levels (percentages) using the `--inference-complexity` flag.

- **Default**: 100 (Full complexity)
- **Supported Values**: 25, 50, 75, 100

**Example**: Run training and then evaluate at 25%, 50%, and 100% complexity:
```bash
python run_adactsi.py --dataset-name pems_point --subdataset-name PEMS04 --inference-complexity 25 50 100
```

---

## 🙏 Acknowledgements

This code builds upon the implementations of [GRIN](https://github.com/Graph-Machine-Learning-Group/grin). We extend our gratitude to their contributions to the field.
