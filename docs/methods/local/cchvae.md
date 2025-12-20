# CCHVAE

**Conditional Counterfactual Hierarchical VAE**

CCHVAE uses a hierarchical variational autoencoder for counterfactual generation.

## Overview

CCHVAE learns a latent representation that enables generating plausible counterfactuals through latent space traversal.

## Usage

```python
from counterfactuals.cf_methods.local_methods.c_chvae import CCHVAE

method = CCHVAE(
    mlmodel=ml_model,
    hyperparams=hyperparams
)

result = method.get_counterfactuals(
    factuals=factuals
)
```

## API Reference

::: counterfactuals.cf_methods.local_methods.c_chvae.c_chvae.CCHVAE
