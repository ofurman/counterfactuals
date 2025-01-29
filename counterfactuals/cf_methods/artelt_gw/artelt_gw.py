import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.cf_methods.warren.cf_dice import DiceExplainer
from counterfactuals.cf_methods.artelt_gw.cf_clustering import cluster_instances
from counterfactuals.cf_methods.artelt_gw.ea_mixedvar_groupcf import (
    compute_mixedvar_groupcf,
)
# from counterfactuals.cf_methods.kanamori.groupcf_kanamori import GroupCF
# from counterfactuals.cf_methods.warren.groupcf_warren import compute_groupcf as compute_groupcf_warren


class ArteltGW(BaseCounterfactual):
    def __init__(self, disc_model, train_dataframe):
        self.disc_model = disc_model
        self.train_dataframe = train_dataframe
        self.features_desc = train_dataframe.columns[:-1]

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        try:
            explanation = compute_mixedvar_groupcf(
                X_orig=X,
                y_target=y_target,
                clf=self.disc_model,
                features_type=["real"] * X.shape[1],
                features_range=[(-1e10, 1.0)] * X.shape[1],
                features_idx_whitelist=list(range(X.shape[1])),
            )
        except Exception as e:
            explanation = None
            print(e)
        return explanation, X, y_origin, y_target
        # return ExplanationResult(
        #     x_cfs=explanation, y_cf_targets=y_target, x_origs=X, y_origs=y_origin
        # )

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ) -> ExplanationResult:
        Xs, ys = dataloader.dataset.tensors
        # create ys_target numpy array same shape as ys but with target class
        # ys_target = np.full(ys.shape, target_class)
        ys_target = np.zeros_like(ys)
        ys_target[:, target_class] = 1
        Xs_cfs = []
        model_returned = []
        Xs = Xs.numpy()

        # Compute individual counterfactuals
        X_train = self.train_dataframe[self.train_dataframe.columns[:-1]]
        y_train = self.train_dataframe[self.train_dataframe.columns[-1]]

        y_train_pred = self.disc_model.predict(X_train.values).numpy()
        ypred_prop = self.disc_model.predict_proba(Xs)[:, 1]
        ypred = self.disc_model.predict(Xs)

        dice_expl = DiceExplainer(self.disc_model, X_train.values, y_train.values)
        X_cfs = []
        X_idx = []
        X_instances = Xs
        for i in range(X_instances.shape[0]):
            try:
                expl = dice_expl.compute_counterfactual(X_instances[i, :], 1)[
                    0
                ].flatten()
                X_cfs.append(expl)
                X_idx.append(i)
            except Exception as e:
                pass

        X_instances = X_instances[
            X_idx, :
        ]  # Remove instances for which no counterfactual was found!
        X_cfs = np.array(X_cfs)

        # Cluster instances
        # ["dbscan-cf", "dbscan-xorig"]
        clustering = cluster_instances(
            X_instances, X_cfs, method="dbscan-xorig"
        ).labels_
        print(f"Clustering: {clustering}")

        # Compute multi-instance counterfactuals
        def compute_multiinstance_cf(X_inst: np.ndarray, multiinst_method="ours"):
            clf = self.disc_model
            y_target = 0
            if multiinst_method == "ours":
                Xs_cfs = compute_mixedvar_groupcf(
                    X_inst,
                    1 - y_target,
                    clf=clf,
                    features_type=["real"] * X_inst.shape[1],
                    features_range=[(0.0, 1.0)] * X_inst.shape[1],
                    features_idx_whitelist=list(range(X_inst.shape[1])),
                )
                return Xs_cfs
            # elif multiinst_method == "warren":
            #     delta_cf, cf_score = compute_groupcf_warren(clf, X_train, y_train, X_inst, 1 - y_target, X_othersamples)
            #     cf_size = len(delta_cf) / X_inst.shape[1]
            #     return delta_cf

            # elif multiinst_method == "kanamori":
            #     expl = GroupCF(clf, self.features_desc, X_train, y_train, 1-y_target)
            #     delta_cf, cf_score = expl.compute_explanation(X_inst)
            #     cf_size = len(list(filter(lambda i: np.abs(delta_cf[i]) >= 1e-5, range(len(delta_cf))))) / X_inst.shape[1]
            #     return delta_cf

        # GLOBAL
        # Xs_cfs = compute_multiinstance_cf(X_instances)

        # GROUP-WISE
        Xs_cfs = []
        for l in tqdm(np.unique(clustering)):  # noqa: E741
            idx = clustering == l
            Xs_cfs_cluster = compute_multiinstance_cf(X_instances[idx, :])
            Xs_cfs.append(Xs_cfs_cluster)

        Xs_cfs = np.concatenate(Xs_cfs, axis=0)

        print(Xs_cfs.shape)
        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs_cfs_full = np.full((Xs.shape[0], Xs.shape[1]), np.nan)
        Xs_cfs_full[X_idx, :] = Xs_cfs
        print(Xs_cfs_full.shape)
        Xs = np.array(Xs)
        ys = np.array(ys)
        ys_target = np.array(ys_target)
        num_clusters = len(np.unique(clustering))
        return Xs_cfs_full, Xs, ys, ys_target, model_returned, num_clusters
        # return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys, x_origs=Xs, y_origs=ys)
