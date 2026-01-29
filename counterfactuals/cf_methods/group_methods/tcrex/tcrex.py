from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for predictions
    torch = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    """Hyperrectangular rule in feature space."""

    bounds: tuple[tuple[float, float], ...]

    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is contained within this rule."""
        for i, (lb, ub) in enumerate(self.bounds):
            if not (lb <= x[i] < ub):
                return False
        return True


@dataclass
class TreeNode:
    """Custom tree node for meta-rule classification."""

    feature: int = -1
    threshold: float = -1.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    prediction: Optional[int] = None
    values: Optional[list[int]] = None

    def is_leaf(self) -> bool:
        """Return True when node is a leaf."""
        return self.prediction is not None


@dataclass
class CRE:
    """Counterfactual Rule Explanation container."""

    target: list[int]
    max_valid_rules: list[Rule]
    meta_rules: list[Rule]
    meta_tree: TreeNode

    def __call__(self, x: np.ndarray) -> tuple[int, Rule]:
        """Return optimal rule index and rule for x."""
        optimal_idx = predict_tree(self.meta_tree, x)
        return optimal_idx, self.max_valid_rules[optimal_idx]


@dataclass
class CategoricalFeatureInfo:
    """Metadata for a one-hot encoded categorical feature."""

    name: str
    feature_indices: list[int]
    n_categories: int

    @property
    def index_set(self) -> set[int]:
        return set(self.feature_indices)


@dataclass
class FeatureConfig:
    """Configuration for numerical and categorical feature indices."""

    n_features: int
    categorical_features: list[CategoricalFeatureInfo] = field(default_factory=list)
    numerical_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        all_categorical_indices: set[int] = set()
        for cat in self.categorical_features:
            all_categorical_indices.update(cat.feature_indices)
        self.numerical_indices = set(range(self.n_features)) - all_categorical_indices

    def get_categorical_for_index(self, idx: int) -> Optional[CategoricalFeatureInfo]:
        """Return the categorical feature containing idx, if any."""
        for cat in self.categorical_features:
            if idx in cat.feature_indices:
                return cat
        return None


def _as_numpy(predictions: Any) -> np.ndarray:
    if torch is not None and isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    return np.asarray(predictions)


def _to_labels(predictions: Any) -> np.ndarray:
    predictions = _as_numpy(predictions)
    if predictions.ndim > 1:
        return predictions.argmax(axis=1)
    return predictions.reshape(-1)


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for labels."""
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    total = len(y)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def split_data(
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data based on a feature threshold."""
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: float,
    current_depth: int,
    allowed_thresholds: list[list[float]],
) -> TreeNode:
    """Build a decision tree with constrained thresholds."""
    filtered_thresholds = []
    for bounds in allowed_thresholds:
        filtered_thresholds.append([t for t in bounds if not np.isinf(t)])

    if len(np.unique(y)) == 1 or current_depth >= max_depth:
        return TreeNode(
            prediction=int(Counter(y).most_common(1)[0][0]),
            values=list(y),
        )

    best_gini = np.inf
    best_feature = -1
    best_threshold = -1.0
    best_split = None

    for feature in range(X.shape[1]):
        for threshold in filtered_thresholds[feature]:
            left_X, left_y, right_X, right_y = split_data(X, y, feature, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            n = len(y)
            weighted_gini = (len(left_y) / n * gini_impurity(left_y)) + (
                len(right_y) / n * gini_impurity(right_y)
            )
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold
                best_split = (left_X, left_y, right_X, right_y)

    if best_feature == -1 or best_split is None:
        return TreeNode(
            prediction=int(Counter(y).most_common(1)[0][0]),
            values=list(y),
        )

    left_X, left_y, right_X, right_y = best_split
    left_node = build_tree(
        left_X, left_y, max_depth, current_depth + 1, allowed_thresholds
    )
    right_node = build_tree(
        right_X, right_y, max_depth, current_depth + 1, allowed_thresholds
    )

    return TreeNode(
        feature=best_feature,
        threshold=best_threshold,
        left=left_node,
        right=right_node,
    )


def predict_tree(node: TreeNode, x: np.ndarray) -> int:
    """Predict class for a single instance using the custom tree."""
    if node.is_leaf():
        return int(node.prediction)
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    return predict_tree(node.right, x)


def extract_rules_from_sklearn_tree(
    tree: DecisionTreeClassifier, n_features: int
) -> list[Rule]:
    """Extract hyperrectangular rules from a sklearn decision tree."""
    tree_ = tree.tree_

    def recurse(node_id: int, conditions: list[list[float]]) -> list[Rule]:
        rules = [Rule(tuple(tuple(b) for b in conditions))]
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            return rules

        feature = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]

        left_conditions = [list(b) for b in conditions]
        left_conditions[feature][1] = min(left_conditions[feature][1], threshold)
        rules.extend(recurse(tree_.children_left[node_id], left_conditions))

        right_conditions = [list(b) for b in conditions]
        right_conditions[feature][0] = max(right_conditions[feature][0], threshold)
        rules.extend(recurse(tree_.children_right[node_id], right_conditions))
        return rules

    initial_conditions = [[float("-inf"), float("inf")] for _ in range(n_features)]
    return recurse(0, initial_conditions)


def extract_rules_from_sklearn_forest(
    forest: RandomForestClassifier, n_features: int
) -> list[Rule]:
    """Extract unique rules from all trees in a random forest."""
    all_rules: list[Rule] = []
    for tree in forest.estimators_:
        all_rules.extend(extract_rules_from_sklearn_tree(tree, n_features))

    seen: set[tuple[tuple[float, float], ...]] = set()
    unique_rules = []
    for rule in all_rules:
        key = tuple(rule.bounds)
        if key not in seen:
            seen.add(key)
            unique_rules.append(rule)
    return unique_rules


def rule_feasibility(rule: Rule, X: np.ndarray) -> float:
    """Compute feasibility as the fraction of points contained by the rule."""
    count = sum(1 for x in X if rule.contains(x))
    return count / len(X)


def rule_accuracy(
    rule: Rule, X: np.ndarray, predictions: np.ndarray, target: list[int]
) -> float:
    """Compute rule accuracy for the target class(es)."""
    in_rule = 0
    correct = 0
    for i, x in enumerate(X):
        if rule.contains(x):
            in_rule += 1
            if predictions[i] in target:
                correct += 1
    return correct / in_rule if in_rule > 0 else 0.0


def is_subrule(rule: Rule, other: Rule) -> bool:
    """Return True when rule is a strict subset of other."""
    if rule.bounds == other.bounds:
        return False
    for (lb1, ub1), (lb2, ub2) in zip(rule.bounds, other.bounds):
        if not (lb2 <= lb1 and ub1 <= ub2):
            return False
    return True


def is_rule_well_formed(rule: Rule, feature_config: FeatureConfig) -> bool:
    """Check categorical well-formedness constraints."""
    for cat_info in feature_config.categorical_features:
        hot_indices = []
        cold_indices = []
        for idx in cat_info.feature_indices:
            lb, ub = rule.bounds[idx]
            if lb >= 0.5:
                hot_indices.append(idx)
            elif ub <= 0.5:
                cold_indices.append(idx)

        if len(hot_indices) > 1:
            return False
        if hot_indices and cold_indices:
            return False
        if len(cold_indices) > cat_info.n_categories - 1:
            return False
    return True


def simplify_rule_categoricals(rule: Rule, feature_config: FeatureConfig) -> Rule:
    """Simplify categorical specifications following Appendix A.2.1."""
    new_bounds = [list(bounds) for bounds in rule.bounds]
    for cat_info in feature_config.categorical_features:
        hot_idx = None
        cold_indices = []
        for idx in cat_info.feature_indices:
            lb, ub = new_bounds[idx]
            if lb >= 0.5:
                hot_idx = idx
            elif ub <= 0.5:
                cold_indices.append(idx)

        if hot_idx is not None:
            for idx in cat_info.feature_indices:
                if idx != hot_idx:
                    new_bounds[idx] = [float("-inf"), float("inf")]
        elif len(cold_indices) == cat_info.n_categories - 1:
            remaining_hot = set(cat_info.feature_indices) - set(cold_indices)
            hot_idx = remaining_hot.pop()
            new_bounds[hot_idx] = [0.5, float("inf")]
            for idx in cold_indices:
                new_bounds[idx] = [float("-inf"), float("inf")]

    return Rule(tuple(tuple(b) for b in new_bounds))


def _modify_bounds_for_subset_check(
    rule: Rule, feature_config: FeatureConfig
) -> list[tuple[float, float]]:
    """Modify bounds for categorical-aware subset checks."""
    modified = [list(bounds) for bounds in rule.bounds]
    for cat_info in feature_config.categorical_features:
        hot_idx = None
        for idx in cat_info.feature_indices:
            lb, _ = modified[idx]
            if lb >= 0.5:
                hot_idx = idx
                break
        if hot_idx is not None:
            for idx in cat_info.feature_indices:
                if idx != hot_idx:
                    lb, ub = modified[idx]
                    if ub > 0.5:
                        modified[idx] = [lb, 0.5]
    return [tuple(b) for b in modified]


def is_subrule_with_categoricals(
    rule: Rule, other: Rule, feature_config: FeatureConfig
) -> bool:
    """Return True when rule is a strict subset of other with categorical semantics."""
    if rule.bounds == other.bounds:
        return False
    rule_modified = _modify_bounds_for_subset_check(rule, feature_config)
    other_modified = _modify_bounds_for_subset_check(other, feature_config)
    for (lb1, ub1), (lb2, ub2) in zip(rule_modified, other_modified):
        if not (lb2 <= lb1 and ub1 <= ub2):
            return False
    return True


def max_valid_rules(
    rules: list[Rule],
    X: np.ndarray,
    predictions: np.ndarray,
    target: list[int],
    tau: float,
    feature_config: Optional[FeatureConfig] = None,
) -> list[Rule]:
    """Find maximal rules that satisfy the accuracy threshold."""
    candidate_rules = [
        rule for rule in rules if rule_accuracy(rule, X, predictions, target) >= tau
    ]

    if feature_config is not None and feature_config.categorical_features:
        candidate_rules = [
            simplify_rule_categoricals(rule, feature_config) for rule in candidate_rules
        ]
        candidate_rules = [
            rule
            for rule in candidate_rules
            if is_rule_well_formed(rule, feature_config)
        ]

    maximal = []
    for i, rule in enumerate(candidate_rules):
        others = candidate_rules[:i] + candidate_rules[i + 1 :]
        is_maximal = True
        for other in others:
            if feature_config is not None and feature_config.categorical_features:
                if is_subrule_with_categoricals(rule, other, feature_config):
                    is_maximal = False
                    break
            else:
                if is_subrule(rule, other):
                    is_maximal = False
                    break
        if is_maximal:
            maximal.append(rule)
    return maximal


def partition_bounds(rules: list[Rule]) -> list[list[float]]:
    """Compute sorted unique bounds per feature dimension."""
    if not rules:
        return []

    n_features = len(rules[0].bounds)
    bounds_per_dim: list[list[float]] = []
    for dim in range(n_features):
        bounds = {float("-inf"), float("inf")}
        for rule in rules:
            lb, ub = rule.bounds[dim]
            bounds.add(lb)
            bounds.add(ub)
        bounds_per_dim.append(sorted(bounds))
    return bounds_per_dim


def induced_grid(rules: list[Rule]) -> list[tuple[tuple[float, float], ...]]:
    """Compute the induced grid partition from rules."""
    bounds_per_dim = partition_bounds(rules)
    consecutive_per_dim = []
    for bounds in bounds_per_dim:
        pairs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        consecutive_per_dim.append(pairs)
    return list(product(*consecutive_per_dim))


def is_valid_grid_cell(
    cell: tuple[tuple[float, float], ...], feature_config: FeatureConfig
) -> bool:
    """Return True when a grid cell is valid for categorical constraints."""
    for cat_info in feature_config.categorical_features:
        hot_count = 0
        cold_count = 0
        for idx in cat_info.feature_indices:
            lb, ub = cell[idx]
            if lb >= 0.5:
                hot_count += 1
            if ub <= 0.5:
                cold_count += 1
        if hot_count > 1:
            return False
        if hot_count == 0 and cold_count < cat_info.n_categories:
            return False
        if hot_count == 0 and cold_count == cat_info.n_categories:
            return False
    return True


def induced_grid_with_categoricals(
    rules: list[Rule], feature_config: Optional[FeatureConfig] = None
) -> list[tuple[tuple[float, float], ...]]:
    """Compute induced grid and filter impossible categorical cells."""
    all_cells = induced_grid(rules)
    if feature_config is None or not feature_config.categorical_features:
        return all_cells
    valid_cells = [
        cell for cell in all_cells if is_valid_grid_cell(cell, feature_config)
    ]
    logger.info(
        "Grid filtering: %s -> %s cells (%s removed)",
        len(all_cells),
        len(valid_cells),
        len(all_cells) - len(valid_cells),
    )
    return valid_cells


def rule_contains_points(rule: Rule, X: np.ndarray) -> np.ndarray:
    """Return the subset of X contained by rule."""
    mask = np.array([rule.contains(x) for x in X])
    return X[mask]


def prototype(
    grid_cell: tuple[tuple[float, float], ...],
    X: np.ndarray,
    pick_arbitrary: bool = False,
) -> np.ndarray:
    """Select a prototype point for a grid cell."""
    rule = Rule(tuple(grid_cell))
    contained = rule_contains_points(rule, X)
    if len(contained) == 0:
        return np.array(
            [
                (lb + ub) / 2
                if not np.isinf(lb) and not np.isinf(ub)
                else (lb if not np.isinf(lb) else ub)
                for lb, ub in grid_cell
            ]
        )
    if pick_arbitrary:
        return contained[np.random.randint(len(contained))]
    return contained.mean(axis=0)


def rule_changes(rule: Rule, x: np.ndarray) -> int:
    """Count number of feature changes needed for x to fit the rule."""
    changes = 0
    for i, (lb, ub) in enumerate(rule.bounds):
        if x[i] < lb or x[i] >= ub:
            changes += 1
    return changes


def rule_cost(rule: Rule, x: np.ndarray, X: np.ndarray) -> float:
    """Compute rule cost for x."""
    return rule_changes(rule, x) - rule_feasibility(rule, X)


def cre_for_point(
    rules: list[Rule], x: np.ndarray, X: np.ndarray, return_index: bool = False
) -> Union[Rule, int]:
    """Return the lowest-cost rule for x."""
    costs = [rule_cost(rule, x, X) for rule in rules]
    idx = int(np.argmin(costs))
    return idx if return_index else rules[idx]


def classify_prototypes(
    prototypes: np.ndarray, rule_assignments: np.ndarray, bounds: list[list[float]]
) -> TreeNode:
    """Build the meta-tree classifier for prototypes."""
    if len(np.unique(rule_assignments)) == 1:
        return TreeNode(
            prediction=int(rule_assignments[0]), values=list(rule_assignments)
        )
    return build_tree(prototypes, rule_assignments, np.inf, 0, bounds)


class TCRExGenerator:
    """T-CREx generator implementing the algorithm from Bewley et al. (2024)."""

    def __init__(
        self,
        rho: float = 0.2,
        tau: float = 0.9,
        use_forest: bool = False,
        surrogate_tree_params: Optional[dict[str, Any]] = None,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        self.rho = rho
        self.tau = tau
        self.use_forest = use_forest
        self.surrogate_tree_params = surrogate_tree_params or {}
        self.feature_config = feature_config

    def grow_surrogate(
        self, X: np.ndarray, predictions: np.ndarray
    ) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
        """Train a tree-based surrogate model."""
        min_samples = max(1, int(self.rho * len(X)))
        params = {"min_samples_leaf": min_samples, "random_state": 42}
        params.update(self.surrogate_tree_params)

        if self.use_forest:
            surrogate = RandomForestClassifier(**params)
        else:
            surrogate = DecisionTreeClassifier(**params)

        surrogate.fit(X, predictions)
        return surrogate

    def generate(
        self,
        target: list[int],
        X: np.ndarray,
        model: Any,
        predict_fn: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> CRE:
        """Generate a counterfactual rule explanation for the target class."""
        predictions = predict_fn(X) if predict_fn is not None else model.predict(X)
        predictions = _to_labels(predictions)

        n_features = X.shape[1]
        if self.feature_config is None:
            self.feature_config = FeatureConfig(n_features=n_features)
        surrogate = self.grow_surrogate(X, predictions)

        if self.use_forest:
            rules = extract_rules_from_sklearn_forest(surrogate, n_features)
        else:
            rules = extract_rules_from_sklearn_tree(surrogate, n_features)

        max_rules = max_valid_rules(
            rules,
            X,
            predictions,
            target,
            self.tau,
            feature_config=self.feature_config,
        )
        if len(max_rules) == 0:
            raise ValueError(
                f"No valid rules found for target {target} with tau={self.tau}. "
                "Try lowering tau or adjusting rho."
            )

        if len(max_rules) == 1:
            return CRE(
                target=target,
                max_valid_rules=max_rules,
                meta_rules=max_rules,
                meta_tree=TreeNode(prediction=0, values=[0]),
            )

        grid = induced_grid_with_categoricals(max_rules, self.feature_config)
        if not grid:
            logger.warning("No valid grid cells after categorical filtering.")
            return CRE(
                target=target,
                max_valid_rules=max_rules,
                meta_rules=max_rules,
                meta_tree=TreeNode(prediction=0, values=[0]),
            )
        prototypes = np.array(
            [prototype(cell, X, pick_arbitrary=False) for cell in grid]
        )
        rule_assignments = np.array(
            [cre_for_point(max_rules, p, X, return_index=True) for p in prototypes]
        )

        bounds = partition_bounds(max_rules)
        meta_tree = classify_prototypes(prototypes, rule_assignments, bounds)

        return CRE(
            target=target,
            max_valid_rules=max_rules,
            meta_rules=max_rules,
            meta_tree=meta_tree,
        )


class TCREx(BaseCounterfactualMethod):
    """T-CREx counterfactual method wrapper."""

    def __init__(
        self,
        target_model: Optional[Any] = None,
        tau: float = 0.9,
        rho: float = 0.02,
        use_forest: bool = False,
        surrogate_tree_params: Optional[dict[str, Any]] = None,
        predict_fn: Optional[Callable[[np.ndarray], Any]] = None,
        categorical_features: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        disc_model = kwargs.pop("disc_model", None)
        target_model = target_model or disc_model
        if target_model is None:
            raise ValueError("TCREx requires a target_model or disc_model.")
        super().__init__(disc_model=target_model, **kwargs)
        self.target_model = target_model
        self.tau = tau
        self.rho = rho
        self.use_forest = use_forest
        self.surrogate_tree_params = surrogate_tree_params or {}
        self.predict_fn = predict_fn
        self.categorical_features_config = categorical_features

        self._generator: Optional[TCRExGenerator] = None
        self._cre: Optional[CRE] = None
        self._target_signature: Optional[tuple[int, ...]] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._feature_config: Optional[FeatureConfig] = None
        self.n_groups_: int = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TCREx":
        """Store training data for rule induction."""
        self._X_train = check_array(X_train)
        self._y_train = _to_labels(y_train)
        self._feature_config = self._build_feature_config(X_train.shape[1])
        return self

    def _build_feature_config(self, n_features: int) -> FeatureConfig:
        if not self.categorical_features_config:
            return FeatureConfig(n_features=n_features)

        cat_infos = []
        for idx, cat_spec in enumerate(self.categorical_features_config):
            if isinstance(cat_spec, dict):
                indices = cat_spec["indices"]
                name = cat_spec.get("name", f"cat_{idx}")
            else:
                indices = list(cat_spec)
                name = f"cat_{idx}"
            cat_infos.append(
                CategoricalFeatureInfo(
                    name=name,
                    feature_indices=list(indices),
                    n_categories=len(indices),
                )
            )
        return FeatureConfig(n_features=n_features, categorical_features=cat_infos)

    def _ensure_cre(self, targets: list[int], X_train: np.ndarray) -> None:
        signature = tuple(sorted(set(targets)))
        if self._generator is None:
            self._generator = TCRExGenerator(
                rho=self.rho,
                tau=self.tau,
                use_forest=self.use_forest,
                surrogate_tree_params=self.surrogate_tree_params,
                feature_config=self._feature_config,
            )

        if self._cre is None or self._target_signature != signature:
            logger.info("Building T-CREx rules for target classes: %s", signature)
            self._cre = self._generator.generate(
                target=list(signature),
                X=X_train,
                model=self.target_model,
                predict_fn=self.predict_fn,
            )
            self._target_signature = signature
            self.n_groups_ = _count_leaves(self._cre.meta_tree)

    def _project_to_rule(self, x: np.ndarray, rule: Rule) -> np.ndarray:
        cf_point = np.array(x, copy=True)
        for i, (lb, ub) in enumerate(rule.bounds):
            if x[i] < lb:
                cf_point[i] = lb
            elif x[i] >= ub:
                if np.isinf(ub):
                    cf_point[i] = x[i]
                else:
                    cf_point[i] = np.nextafter(ub, lb)
        return cf_point

    def explain(
        self,
        X: np.ndarray,
        y_origin: Optional[np.ndarray] = None,
        y_target: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for given instances."""
        X = check_array(X)
        if X_train is not None:
            self._X_train = check_array(X_train)
        if y_train is not None:
            self._y_train = _to_labels(y_train)

        if self._X_train is None:
            raise ValueError("TCREx requires training data via fit() or X_train.")
        if self._feature_config is None:
            self._feature_config = self._build_feature_config(self._X_train.shape[1])

        if y_origin is None:
            y_origin = _to_labels(self.target_model.predict(X))
        if y_target is None:
            raise ValueError("TCREx requires y_target to build target-specific rules.")

        target_labels = _to_labels(y_target)
        self._ensure_cre(target_labels.tolist(), self._X_train)

        cf_points = np.zeros_like(X)
        group_ids = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            rule_idx, rule = self._cre(x)
            group_ids[i] = rule_idx
            cf_points[i] = self._project_to_rule(x, rule)

        return ExplanationResult(
            x_cfs=cf_points,
            y_cf_targets=target_labels,
            x_origs=X,
            y_origs=_to_labels(y_origin),
            logs=None,
            cf_group_ids=group_ids,
        )

    def explain_dataloader(
        self,
        dataloader,
        epochs: int = None,
        lr: float = None,
        patience_eps: Union[float, int] = 1e-5,
        y_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for a DataLoader."""
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for batch in dataloader:
            X_batch, y_batch = batch
            xs.append(_as_numpy(X_batch))
            ys.append(_as_numpy(y_batch))
        X = np.concatenate(xs, axis=0)
        y_origin = _to_labels(np.concatenate(ys, axis=0))

        if y_target is None:
            raise ValueError("TCREx explain_dataloader requires y_target.")

        return self.explain(X=X, y_origin=y_origin, y_target=y_target, **kwargs)


def _count_leaves(node: TreeNode) -> int:
    if node.is_leaf():
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)
