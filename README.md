
# Milestone Assignment 2: Principal Component Analysis (PCA)

This project demonstrates how to use Principal Component Analysis (PCA) on the Breast Cancer dataset from `sklearn.datasets` to identify essential variables and reduce dimensionality to **2 components**. As a bonus, it implements **Logistic Regression** for prediction using both the PCA-reduced features and the full standardized feature set.

## Contents
- `pca_pipeline.py` – Main script that:
  - Loads the Breast Cancer dataset
  - Standardizes features
  - Runs PCA (default: 2 components)
  - Saves a **scree plot**, a **2D scatter** of the first two PCs, and a **CSV** of PCA components
  - Trains **Logistic Regression** on:
    - the 2 PCA components (bonus)
    - the full standardized feature set (for comparison)
  - Prints accuracy and classification report for both models

- `outputs/`
  - `scree_plot.png` – Explained variance ratio of the PCs
  - `pca_scatter.png` – Scatter plot of PC1 vs PC2, colored by target
  - `pca_components.csv` – Table of PC1, PC2, and target labels

## Quick Start

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the PCA pipeline
```bash
python pca_pipeline.py --components 2 --outdir outputs
```

Add `--balanced` to use class-weight balancing in Logistic Regression:
```bash
python pca_pipeline.py --components 2 --outdir outputs --balanced
```

## What the script produces
- **Console output** summarizing:
  - PCA explained variance ratio
  - Logistic Regression accuracy & classification report on:
    - 2 PCA components
    - Full standardized features
- **Files in `outputs/`**:
  - `scree_plot.png`
  - `pca_scatter.png`
  - `pca_components.csv`

## Notes
- The Breast Cancer dataset is binary (malignant vs benign). Using 2 PCs preserves a large portion of variance while providing a compact representation.
- Standardization is important before PCA so each feature contributes equally.
- Logistic Regression gives you a quick, interpretable baseline for prediction.

## How to submit
Compress the project as a ZIP (e.g., `pca_cancer_submission.zip`) and upload it, or push the repo to GitHub and share the link.
