# Counterfactual Explanations Library

This repository is dedicated to the research and development of counterfactual explanations framework. It implements different methods for local, global and groupwise explanations.

<p align="center">
<img src="graphic.svg" alt="drawing" width="800"/>
</p>

# Methods

## Local Methods

| Method | Paper | Framework | DataType | Problem Type |
|---|---|---|---|---|
| artelt | | P | T | C |
| casebased_sace | | P | T | C |
| cegp | | P | T | C |
| cem | | T | T | C |
| ppcef | | P | T | C |
| regression ppcef | | P | T | C |
| pumal | | P | T | C |
| sace | | P | T | C |
| wach | | P | T | C |

## Global Methods

| Method | Paper | Framework | DataType | Problem Type |
|---|---|---|---|---|
| ARES | | P | T | C |
| GLOBE-CE | | P | T | C |
| pumal | | P | T | C |

## Group Methods

| Method | Paper | Framework | DataType | Problem Type |
|---|---|---|---|---|
| glance | | P | T | C |
| pumal | | P | T | C |
| tcrex | | P | T | C |

**Legend:**
*   **Framework**: P - PyTorch, T - TensorFlow, S - scikit-learn
*   **DataType**: T - Tabular, T* - Tabular with continuous features, I - Images
*   **Problem Type**: C - Classification, R - Regression

