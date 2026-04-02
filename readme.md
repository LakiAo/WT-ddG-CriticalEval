# WT-ddG-CriticalEval

**Beyond Random Splits:  
A Critical Evaluation of Graph Learning Models in Predicting Mutation-Induced Drug Resistance**

> Official implementation of the manuscript currently under review

---

## Overview

This project studies the generalization ability of ΔΔG prediction models under a **WT-only setting**,  
where only wild-type protein–ligand complex structures are available.

Unlike most existing methods that rely on both WT and mutant (MT) structures,  
this work evaluates a more realistic scenario for drug resistance prediction.  
This project highlights that random splits may significantly overestimate model generalization.

### 0. Repository Setup

First, clone the repository to your local Linux system:

```bash
git clone https://github.com/LakiAo/WT-ddG-CriticalEval.git
cd WT-ddG-CriticalEval
```

---

### 1. Data Preparation

Before running the model, you need to prepare the required files and datasets.

#### (1) ESMC weights
Download the ESMC weights from the Hugging Face repository:

[https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12)

After downloading, place the files under:

```text
WT-ddG-CriticalEval/graph_model/data/weights
```

#### (2) MdrDB dataset
Download the MdrDB dataset from:

[https://quantum.tencent.com/mdrdb/](https://quantum.tencent.com/mdrdb/)

You need to download the **Single substitution PDB files (.tar.gz)** from the **core set**.

---

### 2. Conda Environment Setup

Then install the required environment. We **strongly recommend using `mamba`** (a faster drop-in replacement for conda) to avoid slow dependency solving:

```bash
conda install -n base -c conda-forge mamba
```

After that, create the environment using:

```bash
mamba env create -f environment.yml
```
### 3. Data Preprocessing and ESMC Embedding

After preparing the dataset and environment, you can run the full preprocessing pipeline, including complex extraction, ESMC embedding generation, and graph construction.

---

#### (1) Extract protein–ligand complexes from MdrDB dataset

```bash
python ./graph_model/datapreprocess/extract_complex.py \
  --data_dir /path/to/data \
  --tsv_path /path/to/MdrDB_CoreSet_release_v1.0.2022.tsv \
  --output_dir /path/to/output
```

- `--data_dir`  
  Root directory of the dataset (e.g., extracted MdrDB core set folders)

- `--tsv_path`  
  Path to the provided TSV file: `MdrDB_CoreSet_release_v1.0.2022.tsv`

- `--output_dir`  
  Directory where processed protein (`.pdb`) and ligand (`.sdf`) files will be saved

---

#### (2) Generate ESMC embeddings

Navigate to the `graph_model` directory:

```bash
cd WT-ddG-CriticalEval/graph_model
```

##### WT embeddings

```bash
python -m datapreprocess.esmc_embedding \
  --data_dir /path/to/output
```

##### MT embeddings

```bash
python -m datapreprocess.esmc_embedding_mt \
  --data_dir /path/to/output \
  --meta_tsv /path/to/MdrDB_CoreSet_release_v1.0.2022.tsv
```

---

#### (3) Graph construction

```bash
python -m datapreprocess.graph_construction \
  --data_dir ./output \
  --output_dir ./pth \
  --replace False \
  --protein_embeddings True
```

---

#### Notes

- `--data_dir` should point to the output directory generated in Step (1)  
- `--meta_tsv` should be the same TSV file used during preprocessing  
- The final processed graph data will be saved in the specified `--output_dir` (e.g., `./pth`)  

### 4. Dataset Preparation and Split

Before training, you need to prepare a unified CSV file and construct dataset splits for cross-validation.

---

#### (1) Generate CSV for graph data

```bash
python ./dataset/extract_csv_uniportid.py \
  --tsv ./MdrDB_CoreSet_release_v1.0.2022.tsv \
  --pth_dir ./pth \
  --out_csv ./dataset/mdrdb_graph_mut_ddg.csv
```

- `--tsv`  
  Path to the metadata file provided by the MdrDB dataset

- `--pth_dir`  
  Directory containing the processed graph files generated in the preprocessing step

- `--out_csv`  
  Output CSV file for downstream dataset splitting

---

#### (2) Construct cross-validation splits

After generating `mdrdb_graph_mut_ddg.csv`, you can construct two types of dataset splits:

- **Random split**
- **UniProt-based split**

##### Random split

```bash
python ./dataset/kfold_random.py \
  --csv ./dataset/mdrdb_graph_mut_ddg.csv \
  --out_dir ./dataset/kfold_random \
  --seed 42 --k 5 --val_frac 0.15 --ddg_abs_max 8.0 --save_outliers
```

##### UniProt-based split

```bash
python ./dataset/kfold_uniprot.py \
  --csv ./dataset/mdrdb_graph_mut_ddg.csv \
  --out_dir ./dataset/kfold_uniprot \
  --seed 42 --k 5 --val_frac 0.15 --ddg_abs_max 8.0 --save_outliers
```

---

### 5. Model Training and Inference

After preparing the dataset splits, you can start training and inference for both **graph_model** and **vector_model**.

---

## 5.1 Graph Model

#### (1) Training

The following script runs five-fold cross-validation on the **UniProt-based split** for all four protein feature modes (`mode1`–`mode4`):

```bash
ROOT_OUT=./kfold_uniprot_runs
SPLIT_ROOT=./dataset/kfold_uniprot

for mode in mode1 mode2 mode3 mode4; do
  for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
    python ./kfold_train.py \
      --graph_dir ./pth \
      --train_csv ${SPLIT_ROOT}/${fold}/train.csv \
      --val_csv   ${SPLIT_ROOT}/${fold}/val.csv \
      --test_csv  ${SPLIT_ROOT}/${fold}/test.csv \
      --out_dir ${ROOT_OUT} \
      --protein_mode ${mode} \
      --kfold_layout \
      --seed 42 \
      --hidden 128 \
      --layers 2 \
      --dropout 0.35 \
      --lr 3e-4 \
      --save_warmup 2 \
      --min_epochs 10 \
      --patience 20 \
      --topk 1 \
      --score_mode pearson \
      --tie_by rmse \
      --std_pred_min 0.10
  done
done > kfold.log 2>&1 &
```

---

#### (2) Inference

```bash
python inference.py \
  --graph_dir ./pth \
  --csv test.csv \
  --ckpt ./runs/best.pt \
  --out_dir ./outdir
```

---

## 5.2 Vector Model

First, switch to the `vector_model` directory:

```bash
cd WT-ddG-CriticalEval/vector_model
```

---

#### (1) Training

```bash
ROOT_OUT=../graph_model/kfold_uniprot_runs
SPLIT_ROOT=../graph_model/kfold_uniprot
DATA_DIR=./output

for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
  python train.py \
    --data_dir ${DATA_DIR} \
    --train_csv ${SPLIT_ROOT}/${fold}/train.csv \
    --val_csv   ${SPLIT_ROOT}/${fold}/val.csv \
    --test_csv  ${SPLIT_ROOT}/${fold}/test.csv \
    --out_dir ${ROOT_OUT} \
    --tag mode0 \
    --kfold_layout \
    --seed 42 \
    --save_warmup 2 \
    --min_epochs 10 \
    --patience 20 \
    --topk 1 \
    --score_mode pearson \
    --tie_by rmse \
    --std_pred_min 0.10
done
```

---

#### (2) Inference

```bash
python inference.py \
  --data_dir ./output \
  --csv test.csv \
  --ckpt ./runs/best.pt \
  --out_dir ./outdir
```

- `--data_dir`  
  Directory containing processed vector features

- `--csv`  
  CSV file for inference samples

- `--ckpt`  
  Path to the trained checkpoint

- `--out_dir`  
  Directory to save prediction results
