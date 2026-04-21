# Training Models

Learn how to train discriminative and generative models for counterfactual generation.

## Discriminative Models (Classifiers)

### MLP Classifier

```python
from cel.models import MLPClassifier

classifier = MLPClassifier(
    num_inputs=n_features,
    num_targets=n_classes,
    hidden_layer_sizes=[128, 64],
    dropout=0.2,
)

classifier.fit(
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=100,
    lr=0.001
)
```

### Logistic Regression

```python
from cel.models import LogisticRegression

classifier = LogisticRegression(num_inputs=n_features, num_targets=n_classes)
classifier.fit(train_loader, test_loader, epochs=50)
```

## Generative Models (Flows)

### Masked Autoregressive Flow (MAF)

```python
from cel.models import MaskedAutoregressiveFlow

flow = MaskedAutoregressiveFlow(
    features=n_features,
    hidden_features=128,
    num_layers=5,
)

flow.fit(
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=200,
    lr=0.0001
)
```

### Other Flows

- **RealNVP**: Affine coupling layers
- **NICE**: Non-volume preserving
- **CNF**: Continuous normalizing flows (for regression)

## Saving and Loading Models

```python
# Save
classifier.save("models/classifier.pt")
flow.save("models/flow.pt")

# Load
classifier.load("models/classifier.pt")
flow.load("models/flow.pt")
```

## Next Steps

- [Generating Counterfactuals](generating-cel.md)
