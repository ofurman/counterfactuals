# Local Methods

Local counterfactual methods generate explanations for **individual instances**. They answer the question: "What minimal changes to this specific input would change the model's prediction?"

## Available Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| [PPCEF](ppcef.md) | Probabilistic counterfactuals with normalizing flows | High plausibility |
| [DiCoFlex](dicoflex.md) | Diverse counterfactuals with flexible constraints | Diversity + constraints |
| [DICE](dice.md) | Diverse counterfactual explanations | Multiple diverse CFs |
| [WACH](wach.md) | Weighted actionable counterfactuals | Actionability focus |
| [SACE](sace.md) | Several SACE variants | Multiple strategies |
| [CEM](cem.md) | Contrastive explanation method | Pertinent negatives |
| [CEGP](cegp.md) | Genetic programming approach | Evolutionary search |
| [CET](cet.md) | Counterfactual explanation trees | Tree-based |
| [CCHVAE](cchvae.md) | Conditional hierarchical VAE | Latent space |
| [Artelt](artelt.md) | Heuristic-based method | Fast computation |

## When to Use Local Methods

Local methods are ideal when you need to:

- Explain a **specific prediction** to a user
- Provide **actionable recourse** for an individual
- Debug model behavior on **particular instances**
- Generate **personalized recommendations**

## Example Usage

```python
from counterfactuals.cf_methods.local_methods import PPCEF

# Initialize method
method = PPCEF(
    gen_model=flow_model,
    disc_model=classifier,
    disc_model_criterion=criterion,
    device="cuda"
)

# Generate counterfactual for a single instance
result = method.explain(
    X=instance,           # Shape: (1, n_features)
    y_origin=0,           # Current prediction
    y_target=1,           # Desired prediction
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    lr=0.01
)

print(f"Original: {instance}")
print(f"Counterfactual: {result.x_cfs}")
```
