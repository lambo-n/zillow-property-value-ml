# Zillow Property Value Prediction (CS471 Final Project)

Regression models that predict a property's assessed value (`taxvaluedollarcnt`, log-transformed) from property features, using the [Zillow Prize Kaggle dataset](https://www.kaggle.com/competitions/zillow-prize-1/overview).

Models compared: linear regression (baseline), k-nearest neighbors, Random Forest, and a multilayer perceptron, each tuned with 5-fold cross-validation.

## Data

The data is not included in this repo. Accept the competition rules on Kaggle, download the files, and unzip them into `data/raw/`.

## Setup

Requires Python 3.14 and an NVIDIA GPU (CUDA 13 driver) for the GPU-accelerated models.

```
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```
.venv/bin/python src/prepare_data.py                           # clean data -> data/processed/
.venv/bin/python src/eda.py                                    # EDA figures -> figures/, results/eda.json
.venv/bin/python -m cuml.accel src/train.py baseline knn rf    # GPU models -> results/
.venv/bin/python src/train.py mlp                              # MLP on CPU -> results/
.venv/bin/python src/plot_results.py                           # predicted-vs-actual, MLP loss curve
cd src && ../.venv/bin/python -m cuml.accel importance.py          # RF permutation importance
```

Training scripts run under `python -m cuml.accel`, which moves supported scikit-learn models (linear regression, k-NN, Random Forest, scaling and encoding) to the GPU without code changes. The MLP runs on the CPU.
