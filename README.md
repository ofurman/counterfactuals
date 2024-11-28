
# PPCEF: Probabilistically Plausible Counterfactual Explanations using Normalizing Flows

This repository is dedicated to the research and development of **PPCEF** (Probabilistically Plausible Counterfactual Explanations using Normalizing Flows), a novel method designed for generating and evaluating counterfactual explanations in machine learning models. The project aims to enhance model interpretability and fairness by providing insights into alternative scenarios that change a model's decision.

<p align="center">
<img src="graphic.svg" alt="drawing" width="800"/>
</p>

## Abstract

We present **PPCEF**, a novel method specifically tailored for generating probabilistically plausible counterfactual explanations. This approach utilizes normalizing flows as density estimators within an unconstrained optimization framework, effectively balancing distance, validity, and probabilistic plausibility in the produced counterfactuals. Our method is notable for its computational efficiency and ability to process large and high-dimensional datasets, making it particularly applicable in real-world scenarios. A key aspect of **PPCEF** is its focus on the plausibility of counterfactuals, ensuring that the generated explanations are coherent and realistic within the context of the original data. Through comprehensive experiments across various datasets and models, we demonstrate that **PPCEF** can successfully generate high-quality counterfactual explanations, highlighting its potential as a valuable tool in enhancing the interpretability and transparency of machine learning systems.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Structure](#code-structure)
- [Data](#data)
- [Experiments](#experiments)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

Counterfactual explanations offer a way to understand machine learning model decisions by explaining what minimal changes would alter a prediction. This project introduces **PPCEF**, a method that leverages optimization and machine learning techniques to generate plausible counterfactuals. Our approach ensures that the generated counterfactuals are not only close to the original data points but also adhere to domain-specific constraints, making them realistic and actionable for decision-makers.

## Prerequisites

This section details the environment setup, including necessary libraries and frameworks. To clone the repository and set up the environment, use the following commands:

```shell
git clone git@github.com:ofurman/counterfactuals.git
cd counterfactuals
./setup_env.sh
```

## Getting Started
The following Python code snippet demonstrates how to use the PPCEF framework for generating counterfactual explanations:

```python
import numpy as np
import torch

from counterfactuals.datasets import MoonsDataset
from counterfactuals.cf_methods.ppcef import PPCEF
from counterfactuals.generative_models import MaskedAutoregressiveFlow
from counterfactuals.discriminative_models import MultilayerPerceptron
from counterfactuals.losses import BinaryDiscLoss
from counterfactuals.metrics import evaluate_cf


dataset = MoonsDataset("../data/moons.csv")
train_dataloader = dataset.train_dataloader(batch_size=128, shuffle=True)
test_dataloader = dataset.test_dataloader(batch_size=128, shuffle=False)

disc_model = MultilayerPerceptron(
    input_size=2, hidden_layer_sizes=[256, 256], target_size=1, dropout=0.2
)
disc_model.fit(
    train_dataloader,
    test_dataloader,
    epochs=5000,
    patience=300,
    lr=1e-3,
)

gen_model = MaskedAutoregressiveFlow(
    features=dataset.X_train.shape[1], hidden_features=8, context_features=1
)
gen_model.fit(train_dataloader, test_dataloader, num_epochs=1000)

cf = PPCEF(
    gen_model=gen_model,
    disc_model=disc_model,
    disc_model_criterion=BinaryDiscLoss(),
    neptune_run=None,
)
cf_dataloader = dataset.test_dataloader(batch_size=1024, shuffle=False)
log_prob_threshold = torch.quantile(gen_model.predict_log_prob(cf_dataloader), 0.25)
deltas, X_orig, y_orig, y_target, logs = cf.explain_dataloader(
    cf_dataloader, alpha=100, log_prob_threshold=log_prob_threshold, epochs=4000
)
X_cf = X_orig + deltas
print(X_cf)
evaluate_cf(
    disc_model=disc_model,
    gen_model=gen_model,
    X_cf=X_cf,
    model_returned=np.ones(X_cf.shape[0]),
    continuous_features=dataset.numerical_features,
    categorical_features=dataset.categorical_features,
    X_train=dataset.X_train,
    y_train=dataset.y_train,
    X_test=X_orig,
    y_test=y_orig,
    median_log_prob=log_prob_threshold,
    y_target=y_target,
)
```
### Jupyter notebook
You can find the example of running algorithm in the jupyter notebook at: [here](notebooks/ppcef.ipynb)

### Pre-trained Models

We publish pre-trained models in the `./models/` directory for immediate use and experimentation.

## Code Structure

The repository is organized as follows to facilitate ease of use and contribution:

```
|── conf/                  # Configuration files
|── data/                  # Datasets
|── models/                # Trained models
|── notebooks/             # Jupyter notebooks for analysis and examples
|── counterfactuals/       # Source code for the framework
|   ├── cf_methods/        # Counterfactual methods
|   ├── discriminative_models/  # Discriminative models for analysis
|   ├── generative_models/      # Generative models for analysis
|   ├── losses/            # Loss functions
|   ├── metrics/           # Evaluation metrics
|   └── pipelines/         # Data and model pipelines
|── README.md              # This document
└── ...
```

## Data

The full data folder can be found under the following link: [Link](data). More details regarding the datasets can be found in the paper in the appendix directory.

## Experiments

To run experiments, prepare the configuration files located in the `counterfactuals/pipelines/conf/` directory:

Execute the following scripts to train models and run experiments:

```shell
python3 counterfactuals/pipelines/run_ppcef_pipeline.py
```

## Citation
```
@inbook{inbook,
  author = {Wielopolski, Patryk and Furman, Oleksii and Stefanowski, Jerzy and Zięba, Maciej},
  year = {2024},
  month = {10},
  pages = {},
  title = {Probabilistically Plausible Counterfactual Explanations with Normalizing Flows},
  isbn = {9781643685489},
  doi = {10.3233/FAIA240584}
}
```
## Contact

In case of questions or comments please contact using LinkedIn: TBA
