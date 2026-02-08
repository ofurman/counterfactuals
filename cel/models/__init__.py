# ruff: noqa: F401

# Base classes
# Classifier models (sklearn-like API)
from cel.models.classifier.logistic_regression import (
    LogisticRegression,
    MultinomialLogisticRegression,
)
from cel.models.classifier.multilayer_perceptron import MLPClassifier

# Legacy models (backward compatibility)
from cel.models.classifier.multilayer_perceptron import (
    MLPClassifier as MultilayerPerceptron,
)
from cel.models.classifier.node.node import NODE
from cel.models.classifier_mixin import ClassifierPytorchMixin

# Generative models
from cel.models.generative.kde import KDE
from cel.models.generative.maf.maf import MaskedAutoregressiveFlow
from cel.models.generative.nice import NICE
from cel.models.generative.real_nvp import RealNVP
from cel.models.pytorch_base import PytorchBase

# Regression models (sklearn-like API)
from cel.models.regression.linear_regression import LinearRegression
from cel.models.regression.mlp_regressor import MLPRegressor
from cel.models.regression_mixin import RegressionPytorchMixin
