import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from cel.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.cf_methods.group_counterfactual_mixin import (
    GroupCounterfactualMixin,
)
from cel.cf_methods.group_methods.glance.dice_wrapper import (
    DiceExplainerWrapper,
)

logger = logging.getLogger(__name__)


class GLANCE(BaseCounterfactualMethod, GroupCounterfactualMixin):
    def __init__(
        self,
        X_test,
        y_test,
        model,
        features,
        k: int = -1,
        s: int = 4,
        m: int = 1,
        target_class: int = 1,
    ) -> None:
        self.features = list(features)
        self.target_name = "target"
        self.features_with_target = self.features + [self.target_name]
        self.model = model
        self.target_class = target_class
        self.X = X_test[y_test != self.target_class]
        self.Y = y_test[y_test != self.target_class]
        self.n = len(self.X)

        self.k = k if k > 0 else self.n  # starting number of a groups
        self.s = s
        self.m = m

    def prep(self, X_train, y_train, method_to_use="dice"):
        self.__cluster()

        if method_to_use == "dice":
            self.explainer = DiceExplainerWrapper(
                X_train,
                y_train,
                self.features_with_target,
                self.model,
                desired_class=self.target_class,
            )

        self.__perform()

    def __cluster(self):
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.X)
        self.clusters = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

    def __np_to_pd(self, arr):
        return pd.DataFrame(arr.reshape(1, -1), columns=self.features)

    def __perform(self) -> None:
        min_c1, min_c2 = (None, None), (None, None)
        best_total_cost = float("inf")

        cent_lab = zip(self.centroids, self.clusters)
        actions = defaultdict(set)  # Tuple -> Set of actions
        merge_history = []
        action_full_history = []
        # First generate counterfactuals m for each cluster center
        for c, c_lab in tqdm(cent_lab):
            assert isinstance(c, np.ndarray), "Centroid must be a numpy array"

            for _m in range(self.m):
                query_instance = self.__np_to_pd(c)
                counterfactual = self.explainer.generate(
                    query_instance, desired_class=self.target_class
                )
                if counterfactual is not None:
                    vec = counterfactual.squeeze() - c
                    actions[(tuple(c), c_lab)].add(tuple(vec))

        while len(actions) > self.s:
            # Then compare the counterfactuals between clusters to find the best pair
            for (c1, c1_lab), c1_actions in actions.items():
                for (c2, c2_lab), c2_actions in actions.items():
                    if c1_lab == c2_lab:
                        continue
                    cluster_dist = self.__calculate_centroid_distance(c1, c2)

                    cosine_dissim = self.__calculate_average_vector_dissimilarity(
                        c1_actions, c2_actions
                    )

                    total_cost = cluster_dist + cosine_dissim

                    if total_cost < best_total_cost:
                        best_total_cost = total_cost
                        min_c1, min_c2 = (c1, c1_lab), (c2, c2_lab)

            # Merge the two clusters
            self.__merge_clusters(min_c1, min_c2, actions)

            # Update the merge history
            merge_history.append((min_c1, min_c2))
            action_full_history.append(actions.copy())

            # Reset the best total cost
            best_total_cost = float("inf")

        self.merge_history = merge_history
        self.action_full_history = action_full_history

        self.final_clusters = list()
        for (cluster, label), actions in actions.items():
            avg_action = np.mean(list(actions), axis=0)
            # vector_from_cluster = avg_action - np.array(cluster)

            self.final_clusters.append((cluster, label, avg_action))

    def get_merge_history(self) -> list:
        return self.merge_history

    def get_action_full_history(self) -> list:
        return self.action_full_history

    def get_counterfactual(
        self,
        query_instance: pd.DataFrame | np.ndarray,
        use_line_search: bool = False,
        line_search_kwargs: dict = {},
    ) -> np.ndarray:
        if isinstance(query_instance, pd.DataFrame):
            query_instance = query_instance.values

        assert isinstance(query_instance, np.ndarray), "Query instance must be a numpy array"

        if len(query_instance.shape) == 2:
            query_instance = query_instance.squeeze(0)

        min_dist = float("inf")
        min_cluster_idx = -1

        for i, (cluster, _, _) in enumerate(self.final_clusters):
            dist = self.__calculate_centroid_distance(cluster, query_instance.tolist())
            if dist < min_dist:
                min_dist = dist
                min_cluster_idx = i

        logger.debug(
            f"Applying action: {self.final_clusters[min_cluster_idx][2]} on {query_instance}"
        )

        translation_vector = self.final_clusters[min_cluster_idx][2]

        if use_line_search:
            counterfatual = self.line_search(
                query_instance,
                translation_vector,
                **line_search_kwargs,
            )
        else:
            counterfatual = query_instance + translation_vector

        return counterfatual

    def line_search(
        self,
        query_instance: np.ndarray,
        vector: np.ndarray,
        alpha: float = 0.05,
        step_size: float = 0.05,
        domain_bounds: tuple = (0.0, 1.0),
    ) -> np.ndarray:
        if len(query_instance.shape) == 2:
            query_instance = query_instance.squeeze(0)

        original_class = self.model.predict_crisp(query_instance)[0]
        predicted_class = original_class.copy()

        while predicted_class == original_class:
            cf = query_instance + alpha * vector
            predicted_class = self.model.predict_crisp(cf)[0]

            alpha += step_size

            # Check if the query instance is within the domain bounds
            if (cf < domain_bounds[0]).any() or (cf > domain_bounds[1]).any():
                logger.info(
                    f"Query instance is out of bounds ({domain_bounds}), breaking the line search: {cf}"
                )
                break

        logger.info(
            f"Line search finished at alpha: {alpha:.2f} with predicted class: {predicted_class} (original: {original_class})"
        )

        return cf

    def __merge_clusters(self, _c1: tuple, _c2: tuple, actions: dict) -> dict:
        """Merge two clusters by averaging their centroids, updating the labels and merging the actions"""
        new_centroid = (np.array(_c1[0]) + np.array(_c2[0])) / 2
        new_label = self.k
        self.k += 1

        c1_action_set = actions[_c1]
        c2_action_set = actions[_c2]

        assert isinstance(c1_action_set, set)
        assert isinstance(c2_action_set, set)

        merged_actions = c1_action_set.union(c2_action_set)

        # Remove from actions
        del actions[_c1]
        del actions[_c2]

        # Add the new cluster
        actions[(tuple(new_centroid), new_label)] = merged_actions

    def __calculate_average_vector_dissimilarity(self, set1: set[list], set2: set[list]) -> float:
        avg_vect_set1 = np.mean([np.array(action) for action in set1], axis=0)
        avg_vect_set2 = np.mean([np.array(action) for action in set2], axis=0)

        cosine_sim = np.dot(avg_vect_set1, avg_vect_set2) / (
            np.linalg.norm(avg_vect_set1) * np.linalg.norm(avg_vect_set2)
        )

        return 1 - cosine_sim

    def __calculate_centroid_distance(self, _c1: list, _c2: list, type: str = "euclidean") -> float:
        c1 = np.array(_c1)
        c2 = np.array(_c2)
        if type == "euclidean":
            return np.linalg.norm(c1 - c2)
        elif type == "manhattan":
            return np.sum(np.abs(c1 - c2))
        elif type == "chebyshev":
            return np.max(np.abs(c1 - c2))
        else:
            raise ValueError("Invalid distance type")

    def get_clusters(self) -> list[tuple]:
        """
        Returns the final clusters as a list of tuples

        Each tuple contains:
        - The centroid of the cluster
        - The label of the cluster
        - The average action of the cluster (centroid + action = counterfactual)
        """
        return self.final_clusters

    def _explain(self, X: np.ndarray) -> np.ndarray:
        X_aff = X.copy()
        for i in range(X_aff.shape[0]):
            X_aff[i] = self.get_counterfactual(X_aff[i])
        return X_aff

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for given instances.

        Args:
            X: Input instances to explain with shape (n_instances, n_features).
            y_origin: Original predictions/labels for X with shape (n_instances,).
            y_target: Desired target predictions/labels with shape (n_instances,).
            X_train: Training data used to fit the underlying explainer.
            y_train: Training labels used to fit the underlying explainer.
            **kwargs: Additional method-specific parameters.

        Returns:
            ExplanationResult: Object containing counterfactuals, targets, originals,
                and any additional logging information.
        """
        if X_train is not None and y_train is not None:
            method_to_use = kwargs.get("method_to_use", "dice")
            self.prep(X_train, y_train, method_to_use=method_to_use)
        elif not hasattr(self, "explainer"):
            raise ValueError("GLANCE must be prepared with training data before explain().")

        X_cfs = self._explain(X)
        return ExplanationResult(
            x_cfs=X_cfs,
            y_cf_targets=y_target,
            x_origs=X,
            y_origs=y_origin,
            logs=None,
            cf_group_ids=np.asarray(self.clusters),
        )

    def explain_dataloader(
        self,
        dataloader,
        epochs: int = None,
        lr: float = None,
        patience_eps: float = 1e-5,
        y_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for data provided via DataLoader.

        Args:
            dataloader: PyTorch DataLoader containing (X, y) pairs.
            epochs: Not used by GLANCE (included for interface compatibility).
            lr: Not used by GLANCE (included for interface compatibility).
            patience_eps: Not used by GLANCE (included for interface compatibility).
            y_target: Optional target labels aligned to the dataloader.
            **kwargs: Additional method-specific parameters.

        Returns:
            ExplanationResult: Object containing generated counterfactuals, targets,
                original instances, and any additional logging information.
        """
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for batch in dataloader:
            batch_x, batch_y = batch[:2]
            xs.append(batch_x.detach().cpu().numpy())
            ys.append(batch_y.detach().cpu().numpy())

        X = np.vstack(xs)
        y_origin = np.concatenate(ys).reshape(-1)

        if y_target is None:
            y_target = np.abs(1 - y_origin)
        else:
            y_target = np.asarray(y_target).reshape(-1)

        return self.explain(
            X=X,
            y_origin=y_origin,
            y_target=y_target,
            X_train=kwargs.get("X_train"),
            y_train=kwargs.get("y_train"),
        )
