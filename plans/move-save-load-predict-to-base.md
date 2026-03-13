# Plan: Move save/load + predict to Base Classes

## Problem

### save/load — 9 redundant overrides

`PytorchBase` already implements `save()` and `load()`. Despite this, **9 subclasses**
re-implement the exact same two-liner:

```python
def save(self, path):
    torch.save(self.state_dict(), path)

def load(self, path):
    self.load_state_dict(torch.load(path))
```

| Class | File | Lines |
|-------|------|-------|
| LogisticRegression | `classifier/logistic_regression.py` | 83–87 |
| MultinomialLogisticRegression | `classifier/logistic_regression.py` | 162–166 |
| MLPClassifier | `classifier/multilayer_perceptron.py` | 114–118 |
| NODE | `classifier/node/node.py` | 129–133 |
| LinearRegression | `regression/linear_regression.py` | 74–78 |
| NICE | `generative/nice.py` | 155–159 |
| RealNVP | `generative/real_nvp.py` | 155–159 |
| MaskedAutoregressiveFlow | `generative/maf/maf.py` | 171–175 |
| KDE | `generative/kde.py` | 272–276 |

One model — **CeFlowGMM** — has a legitimately different save/load that persists
extra metadata (categorical_groups, dequantizer_dividers). This is the only override
that should remain.

**Fix:** Delete the 9 redundant overrides. The inherited `PytorchBase.save/load` does
the same thing.

### predict / predict_proba — 3 distinct patterns

Currently `ClassifierPytorchMixin` and `RegressionPytorchMixin` declare these as
abstract. Every subclass re-implements the same numpy↔tensor conversion boilerplate.
The actual logic falls into **3 patterns**:

#### Pattern A: Binary classifier (LogisticRegression)

```python
def predict(self, X_test):
    # numpy → tensor → forward() → threshold at 0.5 → numpy
    probs = self.forward(X_test)
    return (probs > 0.5).float().view(-1).cpu().numpy()

def predict_proba(self, X_test):
    # numpy → tensor → forward() → stack [1-p, p] → numpy
    probs = self.forward(X_test)
    return torch.hstack([1 - probs, probs]).cpu().numpy()
```

This is a **one-off** because `LogisticRegression.forward()` returns sigmoid output
directly (post-activation), while MLPClassifier/NODE return raw logits and apply
`self.final_activation` separately.

#### Pattern B: Multi-target classifier (MLPClassifier, NODE)

These two are **byte-for-byte identical**:

```python
def predict(self, X_test):
    probs = self.predict_proba(X_test)            # delegates
    predicted = torch.argmax(probs, dim=1)
    return predicted.squeeze().cpu().numpy()

def predict_proba(self, X_test):
    logits = self.forward(X_test)
    probs = self.final_activation(logits)          # Sigmoid or Softmax
    if self.num_targets == 1:
        probs = torch.hstack([1 - probs, probs])   # binary → 2-col
    return probs.cpu().numpy()
```

`MultinomialLogisticRegression` uses the same structure but calls `softmax` inline
instead of `self.final_activation`.

#### Pattern C: Regressor (LinearRegression, MLPRegressor)

```python
def predict(self, X_test):
    preds = self.forward(X_test)
    return preds.cpu().numpy()

def predict_proba(self, X_test):
    raise NotImplementedError
```

### What they all share

Every `predict` / `predict_proba` does:
1. Convert numpy input to `torch.float32` tensor if needed
2. `torch.no_grad()` context manager
3. Call `self.forward(X_test)`
4. Apply some post-processing (activation, argmax, threshold)
5. Convert result back to numpy via `.cpu().numpy()`

Steps 1, 2, and 5 are identical everywhere. Only step 4 varies.

## Proposed Design

### Part 1: Delete redundant save/load overrides

Simply remove the `save` and `load` methods from the 9 classes listed above.
They inherit from `PytorchBase` which already has the identical implementation.

CeFlowGMM keeps its custom override — it saves/loads extra metadata beyond
`state_dict`.

### Part 2: Move predict/predict_proba to ClassifierPytorchMixin

Make `ClassifierPytorchMixin` aware of `forward()`, `num_targets`, and
`final_activation`, then provide concrete implementations:

```python
class ClassifierPytorchMixin(ABC):
    """Mixin providing classification interface for PyTorch models.

    Subclasses must define:
        - forward(x: torch.Tensor) -> torch.Tensor  (raw logits)
        - final_activation: torch.nn.Module          (e.g. Sigmoid, Softmax)
        - num_targets: int
    """

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to float32 tensor if needed."""
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).float()
        return X

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        X_test = self._to_tensor(X_test)
        with torch.no_grad():
            probs = self.predict_proba(X_test)
            if isinstance(probs, np.ndarray):
                probs = torch.from_numpy(probs)
            predicted = torch.argmax(probs, dim=1)
            return predicted.squeeze().cpu().numpy()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        X_test = self._to_tensor(X_test)
        with torch.no_grad():
            logits = self.forward(X_test)
            probs = self.final_activation(logits)
            if self.num_targets == 1:
                probs = torch.hstack([1 - probs, probs])
            return probs.cpu().numpy()
```

**Effect on each classifier:**

| Class | Change |
|-------|--------|
| MLPClassifier | Delete `predict`, `predict_proba` — already uses `self.final_activation` + `self.num_targets`. Works as-is. |
| NODE | Delete `predict`, `predict_proba` — identical to MLP pattern. Works as-is. |
| MultinomialLogisticRegression | Add `self.final_activation = torch.nn.Softmax(dim=1)` in `__init__`. Delete `predict`, `predict_proba`. |
| LogisticRegression | **Requires adjustment.** Currently `forward()` returns post-sigmoid output. Two options: **(a)** change `forward()` to return raw logits and set `self.final_activation = torch.nn.Sigmoid()`, or **(b)** override `predict_proba` to keep current behavior. Option (a) is cleaner and aligns with the other classifiers. |

#### LogisticRegression alignment (option a)

```python
class LogisticRegression(PytorchBase, ClassifierPytorchMixin):
    def __init__(self, num_inputs, num_targets):
        super().__init__(num_inputs, num_targets)
        self.linear = torch.nn.Linear(num_inputs, num_targets)
        self.final_activation = torch.nn.Sigmoid()
        # criterion stays BCEWithLogitsLoss (expects raw logits)

    def forward(self, x):
        return self.linear(x)  # raw logits, no sigmoid
```

This is safe because `fit()` already uses `BCELoss` / `BCEWithLogitsLoss` which
expects either raw logits or probabilities depending on the variant. Switching
`forward()` to return logits means we should also use `BCEWithLogitsLoss` in `fit()`.

### Part 3: Move predict to RegressionPytorchMixin

```python
class RegressionPytorchMixin(ABC):
    """Mixin providing regression interface for PyTorch models."""

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).float()
        return X

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict continuous values.

        Args:
            X_test: Input data of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs).
        """
        X_test = self._to_tensor(X_test)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_test)
            return preds.cpu().numpy()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Not applicable for regression models."""
        raise NotImplementedError("predict_proba is not applicable for regression models")
```

**Effect:** Delete `predict` and `predict_proba` from both `LinearRegression` and
`MLPRegressor`.

### Part 4: Extract `_to_tensor` to avoid duplication

Both mixins need the same numpy→tensor conversion. Extract it to a small standalone
utility or to `PytorchBase`:

```python
# In PytorchBase
@staticmethod
def _to_tensor(X: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert numpy array to float32 tensor if needed."""
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).float()
    return X
```

Both mixins then call `self._to_tensor(X_test)` (resolved via MRO since all
concrete classes inherit from both `PytorchBase` and a mixin).

## Step-by-Step Implementation

### Step 1: Add `_to_tensor` to `PytorchBase`

- File: `models/pytorch_base.py`
- Add static method `_to_tensor`.
- No behavior change to any existing code.

### Step 2: Make `ClassifierPytorchMixin.predict` and `predict_proba` concrete

- File: `models/classifier_mixin.py`
- Remove `@abstractmethod` decorators.
- Add the concrete implementations shown above.
- Add `import torch` and `import numpy as np`.

### Step 3: Align `LogisticRegression.forward()` to return raw logits

- File: `models/classifier/logistic_regression.py`
- Change `forward()` from `torch.sigmoid(self.linear(x))` to `self.linear(x)`.
- Add `self.final_activation = torch.nn.Sigmoid()` to `__init__`.
- Change `self.criterion` from `BCELoss` to `BCEWithLogitsLoss` in `fit()` (since
  `forward()` now returns logits).
- Delete `predict`, `predict_proba`, `save`, `load`.

### Step 4: Align `MultinomialLogisticRegression`

- File: `models/classifier/logistic_regression.py`
- Add `self.final_activation = torch.nn.Softmax(dim=1)` to `__init__`.
- Delete `predict`, `predict_proba`, `save`, `load`.

### Step 5: Delete overrides from MLPClassifier and NODE

- Files: `models/classifier/multilayer_perceptron.py`, `models/classifier/node/node.py`
- Delete `predict`, `predict_proba`, `save`, `load`.
- These already have `self.final_activation` and `self.num_targets`, so the mixin
  implementation works without changes.

### Step 6: Make `RegressionPytorchMixin.predict` and `predict_proba` concrete

- File: `models/regression_mixin.py`
- Remove `@abstractmethod` decorators.
- Add concrete implementations.

### Step 7: Delete overrides from LinearRegression and MLPRegressor

- Files: `models/regression/linear_regression.py`, `models/regression/mlp_regressor.py`
- Delete `predict`, `predict_proba`, `save`, `load`.

### Step 8: Delete save/load from generative models

- Files: `generative/nice.py`, `generative/real_nvp.py`, `generative/maf/maf.py`,
  `generative/kde.py`
- Delete `save` and `load` methods.
- Keep CeFlowGMM's override (it saves extra metadata).

### Step 9: Run tests

```bash
uv run pytest tests/ -x -q
```

All existing tests for classifiers, regressors, and generators should pass
unchanged.

### Step 10: Lint

```bash
uv run ruff check counterfactuals/models/ --fix
uv run ruff format counterfactuals/models/
```

## Summary of Changes per File

| File | Deletions | Additions/Edits |
|------|-----------|-----------------|
| `pytorch_base.py` | — | Add `_to_tensor` static method |
| `classifier_mixin.py` | Remove `@abstractmethod` | Concrete `predict`, `predict_proba` |
| `regression_mixin.py` | Remove `@abstractmethod` | Concrete `predict`, `predict_proba` |
| `logistic_regression.py` | `predict`, `predict_proba`, `save`, `load` × 2 classes | Add `final_activation`, change `forward` to return logits, `BCELoss` → `BCEWithLogitsLoss` |
| `multilayer_perceptron.py` | `predict`, `predict_proba`, `save`, `load` | — |
| `node/node.py` | `predict`, `predict_proba`, `save`, `load` | — |
| `linear_regression.py` | `predict`, `predict_proba`, `save`, `load` | — |
| `mlp_regressor.py` | `predict`, `predict_proba` (no save/load present) | — |
| `nice.py` | `save`, `load` | — |
| `real_nvp.py` | `save`, `load` | — |
| `maf/maf.py` | `save`, `load` | — |
| `kde.py` | `save`, `load` | — |
| `ceflow_gmm.py` | — | Keeps custom override |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `LogisticRegression.forward()` change breaks callers that expect sigmoid output | Grep for direct `.forward()` calls outside `fit()`. The only callers are `predict`/`predict_proba` (being deleted) and `fit()` (switching to `BCEWithLogitsLoss`). Pipeline code calls `.predict()` or `.predict_proba()`, never `.forward()` directly. |
| KDE has custom `load_state_dict` — might interact with base `load()` | Base `load()` calls `load_state_dict()` which KDE already overrides. No conflict — `load()` delegates to the overridden `load_state_dict`. |
| External code calls `model.save()`/`model.load()` | Behavior is identical — same `state_dict` serialization. No change in contract. |
| `_to_tensor` placement in `PytorchBase` — generative mixins don't use it for predict | Generative models have their own `predict_log_proba` with inline conversion. Those can be migrated separately. This plan does not touch generative predict methods. |

## Result

| Metric | Before | After |
|--------|--------|-------|
| `save`/`load` overrides | 9 redundant + 1 legitimate | 0 redundant + 1 legitimate |
| `predict`/`predict_proba` implementations | 6 separate (classifiers) + 2 (regressors) | 1 in ClassifierMixin + 1 in RegressionMixin |
| Lines deleted | ~0 | ~100 |
| New abstractions | — | None (just concrete methods on existing mixins) |
| Breaking changes | — | `LogisticRegression.forward()` returns logits instead of probabilities. Only affects code calling `.forward()` directly, which no pipeline does. |
