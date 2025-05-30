# CIL-Collaborative Filtering

## Overview

This repository provides implementations of various collaborative filtering methods for recommendation systems, including both traditional and neural-based approaches.

### Implemented Models

1. **ALS** – Alternating Least Squares
2. **SVD** – Singular Value Decomposition (via the [Surprise](https://surpriselib.com/) library)
3. **SVD++** – Improved SVD with implicit feedback (Surprise library)
4. **VAE** – Variational Autoencoder
5. **Embeddings** – Custom user/item embedding model
6. **NCF** – Neural Collaborative Filtering (via [LibRecommender](https://github.com/massquantity/LibRecommender))
7. **Attention-based** – Model incorporating attention mechanisms
8. **Modified Deep Matrix** – Deep matrix factorization variant

## Installation
You can set up the environment using [`conda`][1]:
```bash
$ conda env create -f environment.yml
$ source activate cil_env
```
Alternatively, use `pip` to install the dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

To train or evaluate a model, run:

```bash
python run.py \
    --model [als | attention | deepmatrix | embeddings | ncf | svd | svdpp | vae]  
    --mode [train | predict | cv] 
    --fold <n> 
    --val_size <x>
```

### Arguments

* `--model`: Select the algorithm to run.
* `--mode`: Choose the operation mode:

  * `train`: Train the selected model.
  * `predict`: Generate predictions using a trained model.
  * `cv`: Perform cross-validation.
* `--fold <n>`: Number of folds for cross-validation (only used if `--mode=cv`).
* `--val_size <x>`: Proportion of the dataset to use for validation (only used if `--mode=predict`).

## Configuration
You can change the training parameters of each model by altering the `config.py` file.


[1]:https://conda.io/docs/user-guide/install/index.html
