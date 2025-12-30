# Benchmark Results

Comparison of counterfactual methods across datasets.

## Summary

| Method | Validity | Proximity | Sparsity | Plausibility | Speed |
|--------|----------|-----------|----------|--------------|-------|
| PPCEF | 0.95 | 0.82 | 0.71 | **0.89** | Medium |
| DICE | **0.98** | 0.75 | 0.65 | 0.62 | Fast |
| DiCoFlex | 0.94 | 0.80 | 0.73 | 0.85 | Medium |
| GLOBE-CE | 0.91 | **0.88** | **0.78** | 0.71 | Fast |
| ReViCE | 0.93 | 0.84 | 0.74 | 0.82 | Medium |

## Per-Dataset Results

### Adult Dataset

| Method | Validity | Proximity (L2) | Sparsity |
|--------|----------|----------------|----------|
| PPCEF | 0.96 | 0.45 | 3.2 |
| DICE | 0.99 | 0.52 | 4.1 |
| GLOBE-CE | 0.92 | 0.38 | 2.8 |

### COMPAS Dataset

| Method | Validity | Proximity (L2) | Sparsity |
|--------|----------|----------------|----------|
| PPCEF | 0.94 | 0.41 | 2.9 |
| DICE | 0.97 | 0.48 | 3.5 |
| GLOBE-CE | 0.89 | 0.35 | 2.4 |

## Runtime Comparison

| Method | CPU (s/instance) | GPU (s/instance) |
|--------|------------------|------------------|
| PPCEF | 0.15 | 0.02 |
| DICE | 0.08 | N/A |
| GLOBE-CE | 0.05 | 0.01 |

## Reproducing Results

See [Running Benchmarks](running.md) for instructions on reproducing these results.
