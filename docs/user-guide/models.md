# Training Models

Learn how to train discriminative and generative models for counterfactual generation.

## Discriminative Models (Classifiers)

### MLP Classifier

```python
from cel.models.classifiers import MLPClassifier

classifier = MLPClassifier(
    input_dim=n_features,
    hidden_dims=[128, 64],
    output_dim=n_classes
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
from cel.models.classifiers import LogisticRegression

classifier = LogisticRegression(input_dim=n_features, output_dim=n_classes)
classifier.fit(train_loader, test_loader, epochs=50)
```

## Generative Models (Flows)

### Masked Autoregressive Flow (MAF)

```python
from cel.models.generators import MaskedAutoregressiveFlow

flow = MaskedAutoregressiveFlow(
    input_dim=n_features,
    hidden_dims=[128, 128],
    n_layers=5
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

- [Generating Counterfactuals](generating-counterfactuals.md)
