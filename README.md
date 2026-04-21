# GPMA: Gastric Cancer Prognostication Model

This repository contains the code for the **GPMA (Gastric Cancer Prognostication Model)**. The model is built on top of the Mamba architecture and Attention mechanisms (MambaAttn) to achieve highly accurate prognostic predictions and visualizations from Whole Slide Images (WSIs). 

## Project Structure

- `models/`: Contains the definition of the GPMA model (`MambaAttn.py`) and other baseline models.
- `part/`: Contains various network components such as `TokenSelect`, `WTConv2d`, `GLSA`, `DFF`, etc.
- `mamba/`: Contains the core Selective Scan Space State Sequential Model (Mamba) implementation.
- `utils/`: Contains utilities for core training and survival evaluation.
- `train_scripts/`: Bash scripts for running the training pipelines.
- `splits/`: Train/val/test splits for cross-validation.
- `CLAM/`: Contains WSI processing and visualization scripts (e.g., heatmaps, t-SNE).

## Installation

### Prerequisites
- CUDA 11.8 
- Python 3.10

### Environment Setup
1. Create and activate a conda virtual environment:
   ```bash
   conda create -n gpma python=3.10 -y
   conda activate gpma
   ```
2. Install PyTorch 2.0.1:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   pip install packaging
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Mamba kernel:
   ```bash
   cd mamba
   pip install .
   cd ..
   ```

## Usage

### 1. Data Preparation
- Organize your WSIs and extract patch features (e.g., using PLIP, ResNet50, or UNI) into `.pt` files.
- Ensure your extracted features are structured under `dataset_csv/` and match the paths specified in the data splits (`splits/`).

### 2. Training the GPMA Model (Survival Prediction)
We provide training scripts for survival prediction. To train the `mamba_attn` (GPMA) model on Gastric Cancer (STAD):

```bash
bash train_scripts/ATTN_512_survival_k_fold.sh
```

Alternatively, you can run the Python script directly:
```bash
python main_survival.py \
    --drop_out 0.3 \
    --early_stopping \
    --lr 1e-3 \
    --k 5 \
    --k_start 0 \
    --k_end 4 \
    --label_frac 1.0 \
    --max_epochs 100 \
    --model_type mamba_attn \
    --mambamil_layer 2 \
    --mambamil_rate 10 \
    --mambamil_type SRMamba \
    --opt adamw \
    --reg 1e-3 \
    --bag_loss nll_surv \
    --task STAD_survival \
    --split_dir splits/STAD_survival_kfold \
    --in_dim 768
```

### 3. Evaluation
To evaluate a saved checkpoint on a test set, use `model_eval.py`:
```bash
python model_eval.py
```
*Note: You may need to edit the hardcoded checkpoint path and arguments inside `model_eval.py` to point to your specific trained model.*

### 4. Visualization
The `CLAM/` folder contains various scripts to visualize the GPMA model's attention heatmaps and feature distributions:
- **Heatmaps**: `CLAM/create_heatmaps-tsne.py` and `CLAM/create_heatmaps.py`
- **t-SNE & ERF**: `CLAM/create_tsne.py` and `CLAM/create_erf.py`

Example command for heatmap generation:
```bash
python CLAM/create_heatmaps-tsne.py \
    --config config.yaml \
    --model_type mamba_attn \
    --checkpoint /path/to/checkpoint.pth
```
*(Please refer to the specific python scripts for their respective argument requirements.)*

## Acknowledgements
This project is built upon [MambaMIL](https://github.com/isyangshu/MambaMIL) and [CLAM](https://github.com/mahmoodlab/CLAM).
