import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score
from itertools import product

import torch

from counterfactuals.cf_methods.base import BaseCounterfactual


class Hyperrectangle:
    def __init__(self, bounds):
        # Bounds is a list of tuples (lower, upper) for each feature
        self.bounds = bounds

    def contains(self, x):
        return all(l <= xi <= u for xi, (l, u) in zip(x, self.bounds))  # noqa: E741


class CounterfactualRule:
    def __init__(self, hyperrectangle, accuracy, feasibility):
        self.hyperrectangle = hyperrectangle
        self.accuracy = accuracy
        self.feasibility = feasibility


class TCREx(BaseCounterfactual):
    def __init__(self, target_model, tau=0.9, rho=0.02, surrogate_tree_params=None):
        self.target_model = target_model
        self.tau = tau
        self.rho = rho
        self.surrogate_tree_params = surrogate_tree_params or {"max_leaf_nodes": 8}
        self.rules_ = []
        self.metarule_tree_ = None
        self.n_groups_ = 0  # Add attribute to track number of groups

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]

        # Step 1: Train surrogate tree
        surrogate = DecisionTreeClassifier(**self.surrogate_tree_params)
        surrogate.fit(X, y)
        self.surrogate_ = surrogate

        # Step 2: Extract candidate rules (nodes)
        self.rules_ = self._extract_rules(surrogate)

        # Step 3: Filter maximal valid rules
        self.maximal_rules_ = self._filter_maximal_rules()

        # Step 4: Partition input space into grid
        self.grid_cells_ = self._partition_grid()

        # Step 5: Assign optimal rule to each grid cell
        self.cell_rules_ = self._assign_optimal_rules()

        # Step 6: Train metarule tree
        self.metarule_tree_ = self._train_metarule_tree()

        self.n_groups_ = self.metarule_tree_.get_n_leaves()

        return self

    def _extract_rules(self, surrogate):
        # Extract all nodes' hyperrectangles from the surrogate tree
        rules = []
        n_nodes = surrogate.tree_.node_count
        for node_id in range(n_nodes):
            if surrogate.tree_.children_left[node_id] == -1:  # Leaf node
                bounds = self._get_node_bounds(surrogate, node_id)
                feasibility = surrogate.tree_.n_node_samples[node_id] / len(self.X_)
                node_indices = surrogate.apply(self.X_) == node_id
                y_true = self.y_[node_indices]
                y_pred = self.target_model.predict(self.X_[node_indices])
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.numpy().reshape(-1)
                accuracy = accuracy_score(y_true, y_pred)
                rules.append(
                    CounterfactualRule(Hyperrectangle(bounds), accuracy, feasibility)
                )
        return rules

    def _get_node_bounds(self, tree, node_id):
        # Reconstruct bounds from the tree splits
        bounds = [(-np.inf, np.inf)] * self.n_features_
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        node = node_id
        while node != 0:  # Traverse up to root
            parent = np.where(tree.tree_.children_left == node)[0]
            if parent.size > 0:
                parent = parent[0]
                if tree.tree_.feature[parent] != -2:
                    f = tree.tree_.feature[parent]
                    if tree.tree_.threshold[parent] <= bounds[f][1]:
                        bounds[f] = (tree.tree_.threshold[parent], bounds[f][1])
                    node = parent
                    continue
            parent = np.where(tree.tree_.children_right == node)[0]
            if parent.size > 0:
                parent = parent[0]
                if tree.tree_.feature[parent] != -2:
                    f = tree.tree_.feature[parent]
                    if tree.tree_.threshold[parent] >= bounds[f][0]:
                        bounds[f] = (bounds[f][0], tree.tree_.threshold[parent])
                    node = parent
        return bounds

    def _filter_maximal_rules(self):
        # Filter rules by tau and rho, then remove non-maximal
        valid_rules = [
            r
            for r in self.rules_
            if r.accuracy >= self.tau and r.feasibility >= self.rho
        ]
        maximal_rules = []
        for rule in valid_rules:
            is_maximal = True
            for other in valid_rules:
                if rule != other and self._is_subset(
                    rule.hyperrectangle, other.hyperrectangle
                ):
                    is_maximal = False
                    break
            if is_maximal:
                maximal_rules.append(rule)
        return maximal_rules

    def _is_subset(self, hr1, hr2):
        # Check if hr1 is a subset of hr2
        return all(
            l2 <= l1 and u1 <= u2 for (l1, u1), (l2, u2) in zip(hr1.bounds, hr2.bounds)
        )

    def _partition_grid(self):
        # Create grid cells based on maximal rules' bounds
        if (
            not self.maximal_rules_
        ):  # If there are no maximal rules, create at least one grid cell
            return [Hyperrectangle([(-np.inf, np.inf)] * self.n_features_)]

        # Extract all unique bound values per feature
        bounds_per_feature = []
        for d in range(self.n_features_):
            values = set()
            for rule in self.maximal_rules_:
                l, u = rule.hyperrectangle.bounds[d]  # noqa: E741
                values.add(l)
                values.add(u)
            values.add(-np.inf)
            values.add(np.inf)
            bounds_per_feature.append(sorted(values))

        # Create intervals for each feature
        intervals = []
        for bounds in bounds_per_feature:
            feature_intervals = []
            for i in range(len(bounds) - 1):
                feature_intervals.append((bounds[i], bounds[i + 1]))
            intervals.append(feature_intervals)

        # Generate grid cells using Cartesian product
        grid_cells = []
        for cell_intervals in product(*intervals):
            grid_cells.append(Hyperrectangle(list(cell_intervals)))

        return grid_cells

    def _assign_optimal_rules(self):
        # Assign optimal rule to each grid cell (simplified)
        if not self.maximal_rules_:  # Handle case with no maximal rules
            return {}

        # If only one rule exists, assign it to all cells
        if len(self.maximal_rules_) == 1:
            return {cell: self.maximal_rules_[0] for cell in self.grid_cells_}

        # Otherwise compute optimal rule for each cell
        return {cell: self._compute_optimal_rule(cell) for cell in self.grid_cells_}

    def _compute_optimal_rule(self, cell):
        # Compute cost for each rule and select the minimal
        if not self.maximal_rules_:
            return None

        # Get a prototype point from the cell (e.g., midpoint)
        prototype = self._get_prototype(cell)

        costs = []
        for rule in self.maximal_rules_:
            # Calculate sparsity (number of dimensions that need to change)
            sparsity = 0
            for d in range(self.n_features_):
                l, u = rule.hyperrectangle.bounds[d]  # noqa: E741
                if prototype[d] < l or prototype[d] > u:
                    sparsity += 1

            # Cost function: sparsity - feasibility
            cost = sparsity - rule.feasibility
            costs.append(cost)

        return self.maximal_rules_[np.argmin(costs)]

    def _get_prototype(self, cell):
        # Return a representative point for the cell (e.g., midpoint)
        prototype = np.zeros(self.n_features_)
        for d in range(self.n_features_):
            l, u = cell.bounds[d]  # noqa: E741
            # Handle infinite bounds
            if np.isinf(l) and np.isinf(u):
                prototype[d] = 0  # Default to 0 if both bounds are infinite
            elif np.isinf(l):
                prototype[d] = u - 1  # Just inside upper bound
            elif np.isinf(u):
                prototype[d] = l + 1  # Just inside lower bound
            else:
                prototype[d] = (l + u) / 2  # Midpoint
        return prototype

    def _train_metarule_tree(self):
        # Train a decision tree on grid cell prototypes
        if not self.grid_cells_:  # Handle case with no grid cells
            # Create a dummy tree
            meta_tree = DecisionTreeClassifier()
            X_dummy = np.zeros((1, self.n_features_))
            y_dummy = np.zeros(1)
            meta_tree.fit(X_dummy, y_dummy)
            return meta_tree

        X_meta = np.array([self._get_prototype(cell) for cell in self.grid_cells_])
        y_meta = np.array([id(self.cell_rules_[cell]) for cell in self.grid_cells_])

        # Make sure we have unique identifiers for rules
        unique_y = np.unique(y_meta)
        y_labels = np.zeros_like(y_meta)
        for i, val in enumerate(unique_y):
            y_labels[y_meta == val] = i

        meta_tree = DecisionTreeClassifier()
        meta_tree.fit(X_meta, y_labels)
        return meta_tree

    def generate_counterfactual_point(self, x, rule):
        cf_point = np.copy(x)
        for d, (l, u) in enumerate(rule.hyperrectangle.bounds):  # noqa: E741
            if x[d] < l:
                cf_point[d] = l
            elif x[d] > u:
                cf_point[d] = u
        return cf_point

    def explain(self, X):
        # Check if X is a single sample and reshape if needed
        X = check_array(X, ensure_2d=True)

        if self.metarule_tree_ is None:
            # If no metarule tree was trained, return the input
            return X

        leaf_ids = self.metarule_tree_.apply(X)
        cf_points = np.zeros_like(X)

        # Map each leaf to a rule
        prototype_points = np.array(
            [self._get_prototype(cell) for cell in self.grid_cells_]
        )
        prototype_leaf_ids = self.metarule_tree_.apply(prototype_points)

        leaf_to_rule = {}
        for i, leaf_id in enumerate(prototype_leaf_ids):
            if leaf_id not in leaf_to_rule and i < len(self.grid_cells_):
                leaf_to_rule[leaf_id] = self.cell_rules_[self.grid_cells_[i]]

        # Generate counterfactual points
        for i, (x, leaf_id) in enumerate(zip(X, leaf_ids)):
            # If we can't find a rule for this leaf, return the original point
            rule = leaf_to_rule.get(leaf_id)
            if rule:
                cf_points[i] = self.generate_counterfactual_point(x, rule)
            else:
                cf_points[i] = x

        return cf_points

    def explain_dataloader(self, dataloader):
        Xs, ys = dataloader.dataset.tensors
        return self.explain(Xs)
