# Local Methods

Local counterfactual methods generate explanations for **individual instances**. They answer the question: "What minimal changes to this specific input would change the model's prediction?"

## Available Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| [PPCEF](ppcef.md) | Probabilistic counterfactuals with normalizing flows | High plausibility |
| [DICE](dice.md) | Diverse counterfactual explanations | Multiple diverse CFs |
| [WACH](wach.md) | Wachter et al. gradient-based counterfactuals | Classic L2 objective |
| [SACE](sace.md) | Search-based abstract explainer with several variants | Multiple strategies |
| [CEM](cem.md) | Contrastive explanation method (alibi) | Pertinent negatives |
| [CEGP](cegp.md) | Counterfactuals guided by prototypes (alibi) | Prototype-guided search |
| [CCHVAE](cchvae.md) | Conditional hierarchical VAE | Latent-space search |
| [Artelt](artelt.md) | Plausible CFs for linear classifiers | Density-constrained |
| [CADEX](cadex.md) | Constrained adversarial counterfactuals | Categorical/ordinal constraints |
| [CeFlow](ceflow.md) | Normalizing-flow class-mean shift | Latent class means |
| [CEARM](cearm.md) | Bayesian-optimization-based CFs | GP-based search |

## When to Use Local Methods

Local methods are ideal when you need to:

- Explain a **specific prediction** to a user
- Provide **actionable recourse** for an individual
- Debug model behavior on **particular instances**
- Generate **personalized recommendations**

## Example Usage

```python
import torch
from cel.cf_methods.local_methods import WACH

# Initialize method
method = WACH(
    disc_model=classifier,
    disc_model_criterion=torch.nn.BCEWithLogitsLoss(),
    device="cpu",
)

# Generate counterfactuals for a full dataloader
result = method.explain_dataloader(
    dataloader=train_loader,
    epochs=1000,
    lr=5e-4,
    alpha=1.0,
)

print(f"Originals: {result.x_origs.shape}")
print(f"Counterfactuals: {result.x_cfs.shape}")
```
