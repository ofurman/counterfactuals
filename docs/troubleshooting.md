# Troubleshooting Guide

This document covers common issues and their solutions when working with the counterfactuals pipeline.

## Training Flow on Non-Dequantized Data

### Problem Description

When training generative models sucha as normalizing flows on datasets with categorical features, failing to pass the `dequantizer` parameter or passing `None` can lead to incorrect model training and invalid counterfactual generation.

### Root Cause

Generative models like normalizing flows are designed to work with **continuous data**. However, categorical features in tabular datasets are typically encoded as discrete integer values (e.g., 0, 1, 2 for a three-category feature). Training on these discrete values directly can cause:

1. **Distribution mismatch**: The model learns discrete distributions but is evaluated on continuous inputs
2. **Invalid density estimates**: Log-probability calculations assume continuous space
3. **Poor counterfactual quality**: Generated samples may not respect the learned data manifold

### How Dequantization Works

The `GroupDequantizer` transforms discrete categorical features into continuous representations:

```python
# Example: Feature with values [0, 1, 2]
# Original (quantized):     [0, 1, 2, 1, 0, 2]
# After dequantization:     [0.12, 1.34, 2.67, 1.89, 0.23, 2.45]
# After logit transform:    [-2.08, 0.29, 0.98, 0.63, -1.21, 0.82]
```

The process:
1. Adds uniform/gaussian noise to discrete values
2. Scales by the number of unique values (dividers)
3. Applies logit transformation for unconstrained space

### Example Scenario

**Incorrect implementation** (missing dequantizer):

```python
# In run_dice_pipeline.py
dequantizer = GroupDequantizer(dataset.categorical_features_lists)
dequantizer.fit(dataset.X_train)

# WRONG: Passing None or omitting dequantizer
gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer=None)
```

**What happens:**
- Generative model trains on discrete values: `X_train = [[0, 1, 2], [1, 2, 0], ...]`
- During inference, data is dequantized: `X_test = [[0.23, 1.67, 2.89], ...]`
- Model receives continuous values it has never seen during training
- Log-probabilities are computed incorrectly (can be orders of magnitude off)
- Counterfactuals fail validity checks or have poor plausibility scores

**Correct implementation** (line 322 in `run_dice_pipeline.py`):

```python
# Create dequantizer and fit on training data
dequantizer = GroupDequantizer(dataset.categorical_features_lists)
dequantizer.fit(dataset.X_train)

# CORRECT: Pass dequantizer to gen_model creation
gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)
```

**What happens:**
- Inside `train_gen_model`, the dequantizer is passed to `gen_model.fit()`
- Training data is automatically dequantized before each batch
- Model learns continuous distributions that match inference-time data
- Log-probabilities and counterfactuals are computed correctly

### Symptoms

If you encounter these issues, check if dequantizer is being passed correctly:

- **Low validity scores** (<0.3): Counterfactuals don't flip the model's prediction
- **Extreme log-density values**: Values like `-inf`, `nan`, or suspiciously large negative numbers
- **High LOF/Isolation Forest anomaly scores**: Generated counterfactuals are far from training distribution
- **Training/inference metric mismatch**: Model performs well on training data but poorly on test data
- **Error messages** about tensor shape mismatches or invalid probability values

### Verification Steps

1. **Check dequantizer is created and fitted**:

```python
dequantizer = GroupDequantizer(dataset.categorical_features_lists)
dequantizer.fit(dataset.X_train)  # Must be called before use
```

2. **Verify dequantizer is passed to model training**:

```python
gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)
# NOT: gen_model = create_gen_model(cfg, dataset, gen_model_path)
```

3. **Check model is wrapped for inference**:

```python
# After model training, wrap it for proper inverse transformation
gen_model = DequantizationWrapper(gen_model, dequantizer)
```

4. **Inspect training logs**: Look for dequantization operations in the training loop

### Related Files

- `counterfactuals/pipelines/run_dice_pipeline.py:322` - Correct usage example
- `counterfactuals/pipelines/nodes/gen_model_nodes.py:32-74` - Training function that uses dequantizer
- `counterfactuals/dequantization/dequantizer.py` - Dequantizer implementation
- `counterfactuals/dequantization/utils.py` - DequantizationWrapper for inference

### Prevention

- Always create and fit a dequantizer for datasets with categorical features
- Pass the dequantizer explicitly to `create_gen_model`
- Wrap the trained model with `DequantizationWrapper` before inference
- Add assertions to verify dequantizer is not `None` when training flow-based models
- Include dequantization status in experiment logs for debugging
