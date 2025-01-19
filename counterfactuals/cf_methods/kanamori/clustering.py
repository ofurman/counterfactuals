import numpy as np
import time
from utils import flatten, Cost
from ce import ActionExtractor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class Clustering:
    def __init__(
        self,
        mdl,
        X,
        Y=[],
        clustering_object="instance",
        n_clusters=4,
        max_candidates=50,
        print_centers=True,
        tol=1e-6,
        lime_approximation=False,
        n_samples=10000,
        alpha=1.0,
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        target_name="Output",
        target_labels=["Good", "Bad"],
    ):
        self.mdl_ = mdl
        self.extractor_ = ActionExtractor(
            mdl,
            X,
            Y=Y,
            lime_approximation=lime_approximation,
            n_samples=n_samples,
            alpha=alpha,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            max_candidates=max_candidates,
            tol=tol,
            target_name=target_name,
            target_labels=target_labels,
        )
        self.cluster_ = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=0.0001,
            precompute_distances="deprecated",
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs="deprecated",
            algorithm="auto",
        )
        self.n_clusters_ = n_clusters
        self.cost_ = Cost(
            X,
            Y,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            max_candidates=max_candidates,
            tol=tol,
        )
        self.print_centers_ = print_centers
        self.clustering_object_ = clustering_object
        if clustering_object == "action":
            self.neighbors_ = KNeighborsClassifier(
                n_neighbors=1,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=None,
            )
        self.lime_approximation_ = lime_approximation

        self.feature_names_ = (
            feature_names
            if len(feature_names) == X.shape[1]
            else ["x_{}".format(d) for d in range(X.shape[1])]
        )
        self.feature_types_ = (
            feature_types
            if len(feature_types) == X.shape[1]
            else ["C" for d in range(X.shape[1])]
        )
        self.feature_categories_ = feature_categories
        self.feature_categories_flatten_ = flatten(feature_categories)
        self.feature_constraints_ = (
            feature_constraints
            if len(feature_constraints) == X.shape[1]
            else ["" for d in range(X.shape[1])]
        )
        self.target_name_ = target_name
        self.target_labels_ = target_labels
        self.tol_ = tol
        self.infeasible_ = False
        self.feature_categories_inv_ = []
        for d in range(X.shape[1]):
            g = -1
            if self.feature_types_[d] == "B":
                for i, cat in enumerate(self.feature_categories_):
                    if d in cat:
                        g = i
                        break
            self.feature_categories_inv_.append(g)

    def fit(
        self,
        X,
        max_change_num=4,
        cost_type="TLPS",
        gamma=1.0,
        dataset_name="",
        solver="cplex",
        time_limit=180,
        log_stream=False,
        mdl_name="",
        log_name="",
        init_sols={},
        verbose=False,
    ):
        self.X_ = X
        self.N_, self.D_ = X.shape
        self.max_change_num_ = max_change_num
        self.cost_type_ = cost_type
        self.gamma_ = gamma
        self.time_limit_ = time_limit
        start = time.perf_counter()

        if self.clustering_object_ == "instance":
            self.cluster_ = self.cluster_.fit(X)
            self.centers_ = self.cluster_.cluster_centers_
            K = self.cluster_.predict(X)
        elif self.clustering_object_ == "action":
            A = np.zeros([self.N_, self.D_])
            for n in range(self.N_):
                action_dict = self.extractor_.extract(
                    X[n].reshape(1, -1),
                    max_change_num=self.max_change_num_,
                    cost_type=self.cost_type_,
                    tradeoff_parameter=self.gamma_,
                    solver=solver,
                    time_limit=self.time_limit_,
                )
                A[n] = action_dict["action"]
            self.cluster_ = self.cluster_.fit(A)
            self.centers_ = self.cluster_.cluster_centers_
            K = self.cluster_.predict(A)
            self.neighbors_ = self.neighbors_.fit(X, K)
            # if(len(dataset_name)!=0): self.scatter_decomposed(X, A, K, filename=dataset_name)

        self.actions_ = []
        for k in range(self.n_clusters_):
            X_k = X[K == k]
            action_dict = self.extractor_.extract(
                X_k,
                max_change_num=self.max_change_num_,
                cost_type=self.cost_type_,
                tradeoff_parameter=self.gamma_,
                solver=solver,
                time_limit=self.time_limit_,
            )
            action_dict["center"] = (
                self.centers_[k]
                if self.clustering_object_ == "instance"
                else X_k.mean(axis=0)
            )
            # action_dict['center'] = self.centers_[k]
            self.actions_ += [action_dict]
        self.time_ = time.perf_counter() - start

        return self

    def feasify(self, a, x):
        for d in [d for d in range(self.D_) if self.feature_types_[d] == "B"]:
            x_d = x[d] + a[d]
            if x_d not in [0, 1]:
                # print(self.feature_names_[d], x_d)
                a[d] = 0
        for G in self.feature_categories_:
            x_G = x[G] + a[G]
            if x_G.sum() != 1:
                # for d in G: print(self.feature_names_[d], x[d]+a[d])
                a[G] = 0
        return a

    def predict(self, X):
        K = (
            self.cluster_.predict(X)
            if self.clustering_object_ == "instance"
            else self.neighbors_.predict(X)
        )
        A = [self.actions_[k]["action"] for k in K]
        return np.array([self.feasify(a, x) for a, x in zip(A, X)])

    def predict_random(self, X):
        K = (
            self.cluster_.predict(X)
            if self.clustering_object_ == "instance"
            else self.neighbors_.predict(X)
        )
        K_random = [
            np.random.choice([k_ for k_ in range(self.n_clusters_) if k_ != k])
            for k in K
        ]
        A = [self.actions_[k]["action"] for k in K_random]
        return np.array([self.feasify(a, x) for a, x in zip(A, X)])

    def cost(self, X, cost_type="TLPS", random=False):
        A = self.predict_random(X) if random else self.predict(X)
        return np.array(
            [self.cost_.compute(x, a, cost_type=cost_type) for x, a in zip(X, A)]
        ).mean()

    def loss(self, X, target=0, random=False):
        A = self.predict_random(X) if random else self.predict(X)
        return (self.mdl_.predict(X + A) != target).mean()

    def scatter_decomposed(self, X, A, K, filename=""):
        plt.figure(figsize=(10, 8))
        if len(filename) != 0:
            plt.suptitle(filename)

        plt.subplot(2, 2, 1)
        method = "PCA"
        decom = PCA(
            n_components=2,
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=1,
        )
        x = decom.fit_transform(A)
        plt.title("Actions in 2D ({})".format(method))
        plt.scatter(x[:, 0], x[:, 1], c=K)

        plt.subplot(2, 2, 2)
        method = "t-SNE"
        decom = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            n_iter=1000,
            n_iter_without_progress=300,
            min_grad_norm=1e-07,
            metric="euclidean",
            init="random",
            verbose=0,
            random_state=1,
            method="barnes_hut",
            angle=0.5,
            n_jobs=None,
        )
        x = decom.fit_transform(A)
        plt.title("Actions in 2D ({})".format(method))
        plt.scatter(x[:, 0], x[:, 1], c=K)

        plt.subplot(2, 2, 3)
        method = "PCA"
        decom = PCA(
            n_components=2,
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=1,
        )
        x = decom.fit_transform(X)
        plt.title("Instances in 2D ({})".format(method))
        plt.scatter(x[:, 0], x[:, 1], c=K)

        plt.subplot(2, 2, 4)
        method = "t-SNE"
        decom = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            n_iter=1000,
            n_iter_without_progress=300,
            min_grad_norm=1e-07,
            metric="euclidean",
            init="random",
            verbose=0,
            random_state=1,
            method="barnes_hut",
            angle=0.5,
            n_jobs=None,
        )
        x = decom.fit_transform(X)
        plt.title("Instances in 2D ({})".format(method))
        plt.scatter(x[:, 0], x[:, 1], c=K)

        plt.tight_layout()
        if len(filename) == 0:
            plt.show()
        else:
            plt.savefig("res/plot_{}.png".format(filename))
        plt.clf()
        return

    def __str__(self):
        s = ""
        for k, action_dict in enumerate(self.actions_):
            s += "- Cluster {}: \n".format(k + 1)
            s += "\t* Action [{}: {} -> {}] (Acc. = {}/{} = {:.1%} / MeanCost = {:.3}):\n".format(
                self.target_name_,
                self.target_labels_[1],
                self.target_labels_[0],
                action_dict["active"].sum(),
                action_dict["sample"],
                action_dict["active"].sum() / action_dict["sample"],
                action_dict["cost"].sum() / action_dict["sample"],
            )
            action = action_dict["action"]
            for i, d in enumerate(np.where(abs(action) > 1e-8)[0]):
                g = self.feature_categories_inv_[d]
                if g == -1:
                    if self.feature_types_[d] == "C":
                        s += "\t\t* {}: {:+.4f}\n".format(
                            self.feature_names_[d], action[d]
                        )
                    elif self.feature_types_[d] == "B":
                        if action[d] == -1:
                            s += "\t\t* {}: True -> False\n".format(
                                self.feature_names_[d],
                            )
                        else:
                            s += "\t\t* {}: False -> True\n".format(
                                self.feature_names_[d],
                            )
                    else:
                        s += "\t\t* {}: {:+}\n".format(
                            self.feature_names_[d], action[d].astype(int)
                        )
                else:
                    if action[d] == -1:
                        continue
                    cat_name, nxt = self.feature_names_[d].split(":")
                    cat = self.feature_categories_[g]
                    prv = self.feature_names_[
                        cat[np.where(action[cat] == -1)[0][0]]
                    ].split(":")[1]
                    s += '\t\t* {}: "{}" -> "{}"\n'.format(cat_name, prv, nxt)
            if self.print_centers_:
                s += "\t* Center:\n"
                for d, x_d in enumerate(action_dict["center"]):
                    # s += '\t\t* {}: {}\n'.format(self.feature_names_[d], x_d)
                    g = self.feature_categories_inv_[d]
                    if g == -1:
                        if self.feature_types_[d] == "B":
                            s += "\t\t* {}: {:.1%}\n".format(
                                self.feature_names_[d], x_d
                            )
                        else:
                            s += "\t\t* {}: {:.2f}\n".format(
                                self.feature_names_[d], x_d
                            )
                for G in self.feature_categories_:
                    group, _ = self.feature_names_[G[0]].split(":")
                    s += "\t\t* {}:\n".format(group)
                    for d in G:
                        x_d = action_dict["center"][d]
                        if x_d < 1e-8:
                            continue
                        _, cat = self.feature_names_[d].split(":")
                        s += "\t\t\t* {}: {:.1%}\n".format(cat, x_d)
        return s

    def to_markdown(self):
        s = "| | HowToChange |\n"
        s += "| --- | --- |\n"
        for k, action_dict in enumerate(self.actions_):
            a = action_dict["action"]
            acc = action_dict["active"].sum() / action_dict["sample"]
            cost = action_dict["cost"].sum() / action_dict["sample"]
            s += "| Action {} | ".format(k + 1)
            for d in np.where(abs(a) > 1e-8)[0]:
                g = self.feature_categories_inv_[d]
                if g == -1:
                    if self.feature_types_[d] == "C":
                        s += "{}: {:+.4f} <br>".format(self.feature_names_[d], a[d])
                    elif self.feature_types_[d] == "B":
                        if a[d] == -1:
                            s += "{}: True -> False <br> ".format(
                                self.feature_names_[d],
                            )
                        else:
                            s += "{}: False -> True <br> ".format(
                                self.feature_names_[d],
                            )
                    else:
                        s += "{}: {:+} <br>".format(
                            self.feature_names_[d], a[d].astype(int)
                        )
                else:
                    if a[d] == -1:
                        continue
                    cat_name, nxt = self.feature_names_[d].split(":")
                    cat = self.feature_categories_[g]
                    prv = self.feature_names_[cat[np.where(a[cat] == -1)[0][0]]].split(
                        ":"
                    )[1]
                    s += '{}: "{}" -> "{}" <br> '.format(cat_name, prv, nxt)
            s += "(Acc: {:.1%} / Cost: {:.3}) |\n".format(acc, cost)

        s += "\n| Feature "
        for k in range(self.n_clusters_):
            s += "| Cluster {} ".format(k + 1)
        s += "|\n"
        s += "| --- |" + " ---: |" * self.n_clusters_ + "\n"
        X = np.array([action_dict["center"] for action_dict in self.actions_])
        for d, X_d in enumerate(X.T):
            s += "| {} ".format(
                self.feature_names_[d] + ":True"
                if self.feature_types_[d] == "B"
                and self.feature_categories_inv_[d] == -1
                else self.feature_names_[d]
            )
            for x_d in X_d:
                if self.feature_types_[d] == "B":
                    s += "| {:.1%} ".format(x_d)
                else:
                    s += "| {:.2f} ".format(x_d)
            s += "|\n"
        return s


def _check(dataset="h", N=10):
    from sklearn.linear_model import LogisticRegression

    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.neural_network import MLPClassifier
    from utils import DatasetHelper

    np.random.seed(0)

    GAMMA = 0.7

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    mdl = mdl.fit(X_tr, y_tr)
    X = X_ts[mdl.predict(X_ts) == 1]

    # for d in range(X.shape[1]): print(D.feature_names[d], D.feature_types[d], D.feature_constraints[d], mdl.coef_[0][d])
    print("# Clustering Actionable Recourse Summary")
    print("* Dataset:", D.dataset_fullname)
    for d in range(X.shape[1]):
        print(
            "\t* x_{:<2}: {} ({}:{})".format(
                d + 1, D.feature_names[d], D.feature_types[d], D.feature_constraints[d]
            )
        )
    print()

    clustering = Clustering(
        mdl,
        X_tr,
        Y=y_tr,
        clustering_object="instance",
        n_clusters=4,
        print_centers=False,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )
    print("## Learning Clusterwise AReS")
    clustering = clustering.fit(
        X[:N], max_change_num=4, cost_type="MPS", gamma=GAMMA, time_limit=60
    )
    print("- Parameters:")
    print("\t- clustering object: {}".format(clustering.clustering_object_))
    print("\t- num. of clusters: {}".format(clustering.n_clusters_))
    print("\t- gamma: {}".format(clustering.gamma_))
    print()
    print("### Learned Clusterwise AReS")
    print(clustering)
    print("### Score:")
    cost = clustering.cost(X[:N], cost_type="MPS")
    loss = clustering.loss(X[:N])
    print("- cost: {}".format(cost))
    print("- loss: {}".format(loss))
    print("- objective: {}".format(cost + GAMMA * loss))


if __name__ == "__main__":
    _check(dataset="d", N=10)
