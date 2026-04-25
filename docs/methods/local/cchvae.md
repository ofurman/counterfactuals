# CCHVAE

**C-CHVAE: Counterfactuals via Conditional Hierarchical Variational Autoencoder**

CCHVAE samples plausible counterfactuals by searching on a growing hypersphere in the latent space of a VAE.

## Overview

CCHVAE encodes a factual into latent space, samples candidates on an expanding p-norm hypersphere shell, decodes them, enforces categorical encoding constraints, and returns the closest candidate that flips the predicted label.

## Usage

```python
from cel.cf_methods.local_methods import CCHVAE

hyperparams = {
    "data_name": "adult",
    "n_search_samples": 300,
    "p_norm": 2,
    "step": 0.1,
    "max_iter": 2000,
    "clamp": True,
    "binary_cat_features": True,
    "vae_params": {
        "layers": [20, 16, 8],
        "train": True,
        "kl_weight": 0.3,
        "lambda_reg": 1e-6,
        "epochs": 5,
        "lr": 1e-3,
        "batch_size": 32,
    },
}

method = CCHVAE(mlmodel=ml_model, hyperparams=hyperparams)

df_cfs = method.get_counterfactuals(factuals=factuals)
```

## API Reference

::: cel.cf_methods.local_methods.c_chvae.c_chvae.CCHVAE
