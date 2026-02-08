import logging
import time

import numpy as np

from cel.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from cel.cf_methods.local_methods.cet.ce import ActionExtractor
from cel.cf_methods.local_methods.cet.rule_miner import (
    FeatureDiscretizer,
    FrequentRuleMiner,
)
from cel.cf_methods.local_methods.cet.utils import (
    Cost,
    LimeEstimator,
    flatten,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Node:
    def __init__(self, parent, branch=None):
        self.parent = parent
        self.branch = branch
        self.depth = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = 0.5
        self.action = None
        self.cost = 0
        self.active = 0
        self.objective = 0
        self.sample = None
        self.sample_rule = None
        self.sample_indices = None
        self.is_infeasible = False
        self.idx = None
        self.is_changed = False

    def n_sample(self):
        return 0 if self.sample is None else self.sample.shape[0]

    def is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, x):
        if self.is_leaf():
            return self.action
        else:
            if x[self.feature] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    def apply(self, x):
        if self.is_leaf():
            return self.idx
        else:
            if x[self.feature] <= self.threshold:
                return self.left.apply(x)
            else:
                return self.right.apply(x)

    def getRatio(self):
        return self.active / self.sample.shape[0]

    def getObjective(self):
        if self.is_leaf():
            return self.objective / self.sample.shape[0]
        else:
            return (self.left.getObjective() + self.left.getObjective()) / self.sample.shape[0]

    def getLoss(self):
        if self.is_leaf():
            return (self.sample.shape[0] - self.active) / self.sample.shape[0]
        else:
            return (self.left.getLoss() + self.left.getLoss()) / self.sample.shape[0]

    def setIndex(self, idx):
        self.idx = idx
        return self

    def setSample(self, X):
        self.sample = X
        return self

    def setSampleRule(self, X_rule):
        self.sample_rule = X_rule
        return self

    def setSampleIndices(self, X_index):
        self.sample_indices = X_index
        return self

    def setAction(self, action=None, cost=0, active=0, objective=0, is_infeasible=False):
        self.action = action
        self.cost = cost
        self.active = active
        self.objective = objective
        self.is_infeasible = is_infeasible
        return self

    def setRule(self, r):
        self.feature = r
        return self

    def setDepth(self, depth):
        self.depth = depth
        return self


class DummyNode:
    def __init__(self):
        self.left = None
        self.idx = -1
        self.is_changed = False

    def is_leaf(self):
        return self.left is None

    def predict(self, x):
        return self.left.predict(x)

    def apply(self, x):
        return self.left.apply(x)


class CounterfactualExplanationTree(BaseCounterfactualMethod, LocalCounterfactualMixin):
    def __init__(
        self,
        mdl,
        X,
        Y=[],
        max_iteration=1000,
        max_depth=3,
        min_samples_leaf=1,
        remain_redundant_leaf=False,
        max_candidates=50,
        tol=1e-6,
        use_mined_rules=False,
        minsup=0.5,
        discretization_bins=5,
        lime_approximation=False,
        n_samples=10000,
        alpha=1.0,
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        target_name="Output",
        target_labels=["Good", "Bad"],
        device: str | None = None,
        **kwargs,
    ):
        # Initialize base/mixin behavior
        super().__init__(disc_model=mdl, device=device)
        self.mdl_ = mdl
        self.extractor_ = ActionExtractor(
            mdl,
            X,
            Y=Y,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            max_candidates=max_candidates,
            tol=tol,
            target_name=target_name,
            target_labels=target_labels,
            lime_approximation=lime_approximation,
            n_samples=n_samples,
            alpha=alpha,
        )
        self.cost_ = Cost(
            X,
            Y,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            max_candidates=max_candidates,
            tol=tol,
        )

        self.lime_approximation_ = lime_approximation
        if lime_approximation:
            self.lime_ = LimeEstimator(
                mdl,
                X,
                n_samples=n_samples,
                feature_types=feature_types,
                feature_categories=feature_categories,
                alpha=alpha,
            )

        self.max_iteration_ = max_iteration
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.remain_redundant_leaf_ = remain_redundant_leaf
        self.feature_names_ = (
            feature_names
            if len(feature_names) == X.shape[1]
            else ["x_{}".format(d) for d in range(X.shape[1])]
        )
        self.feature_types_ = (
            feature_types if len(feature_types) == X.shape[1] else ["C" for d in range(X.shape[1])]
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

        if use_mined_rules:
            self.discretizer_ = FrequentRuleMiner(minsup=minsup, discretization=True)
            self.discretizer_ = self.discretizer_.fit(
                X,
                feature_names=feature_names,
                feature_types=feature_types,
                discretization_bins=discretization_bins,
            )
            self.rule_names_ = self.discretizer_.rule_names_
            self.R_ = len(self.rule_names_)
            self.rule_length_ = self.discretizer_.L_
        else:
            self.discretizer_ = FeatureDiscretizer(bins=discretization_bins, onehot=False)
            self.discretizer_ = self.discretizer_.fit(
                X, feature_names=feature_names, feature_types=feature_types
            )
            self.rule_names_ = self.discretizer_.feature_names
            self.R_ = len(self.rule_names_)
            self.rule_length_ = np.ones(self.R_)
        self.rule_probability_ = (1 / self.rule_length_) / (1 / self.rule_length_).sum()

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
        X_train: np.ndarray | None = None,
        **kwargs,
    ) -> ExplanationResult:
        """Wrapper to produce ExplanationResult for compatibility.

        This uses the existing get_counterfactuals method and returns an
        ExplanationResult dataclass.
        """
        # If training data provided, fit (keeps backward compatibility)
        if X_train is not None:
            self.fit(X_train)

        x_cfs = self.get_counterfactuals(X)
        y_target_arr = (
            np.array(y_target) if y_target is not None else np.zeros((X.shape[0],), dtype=int)
        )
        y_origin_arr = (
            np.array(y_origin) if y_origin is not None else np.zeros((X.shape[0],), dtype=int)
        )

        return ExplanationResult(
            x_cfs=np.array(x_cfs),
            y_cf_targets=y_target_arr,
            x_origs=np.array(X),
            y_origs=y_origin_arr,
            logs=None,
        )

    def generateTree(self):
        root = Node(self.dummy_, branch="left")
        root = root.setDepth(0)
        stack = [root]
        while len(stack) > 0:
            node = stack.pop()
            node.is_changed = True
            if node.depth < self.max_depth_:
                r = self.selectRule()
                node = node.setRule(r)
                right = Node(node, branch="right")
                left = Node(node, branch="left")
                left = left.setDepth(node.depth + 1)
                right = right.setDepth(node.depth + 1)
                node.right = right
                stack.append(right)
                node.left = left
                stack.append(left)
        return root

    def extractAction(self, X, X_indices):
        action_dict = self.extractor_.extract(
            X,
            lime_coefs=self.lime_coefs_[X_indices] if self.lime_approximation_ else None,
            lime_intercepts=self.lime_intercepts_[X_indices] if self.lime_approximation_ else None,
            max_change_num=self.max_change_num_,
            cost_type=self.cost_type_,
            tradeoff_parameter=self.gamma_,
            solver="cplex",
            time_limit=self.time_limit_,
        )
        if action_dict["feasible"]:
            return (
                action_dict["action"],
                action_dict["cost"].sum(),
                action_dict["active"].sum(),
                action_dict["objective"],
                False,
            )
        else:
            return None, 0, 0, 0, True

    def optimizeTree(self):
        root = self.dummy_.left
        root = root.setSample(self.X_)
        root = root.setSampleRule(self.X_rule_)
        root = root.setSampleIndices(np.arange(self.N_))
        root = root.setDepth(0)
        parent = root.parent
        stack = [root]
        idx = 0
        while len(stack) > 0:
            node = stack.pop()
            if node.is_leaf():
                if node.is_changed:
                    action, cost, active, objective, is_infeasible = self.extractAction(
                        node.sample, node.sample_indices
                    )
                    node = node.setAction(
                        action=action,
                        cost=cost,
                        active=active,
                        objective=objective,
                        is_infeasible=is_infeasible,
                    )
                node = node.setIndex(idx)
                idx += 1
            else:
                is_left = node.sample_rule[:, node.feature] <= node.threshold
                is_right = node.sample_rule[:, node.feature] > node.threshold
                X_left = node.sample[is_left]
                X_rule_left = node.sample_rule[is_left]
                X_indices_left = node.sample_indices[is_left]
                X_right = node.sample[is_right]
                X_rule_right = node.sample_rule[is_right]
                X_indices_right = node.sample_indices[is_right]
                if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                    node.left = node.left.setSample(X_left)
                    node.left = node.left.setSampleRule(X_rule_left)
                    node.left = node.left.setSampleIndices(X_indices_left)
                    node.left = node.left.setDepth(node.depth + 1)
                    node.right = node.right.setSample(X_right)
                    node.right = node.right.setSampleRule(X_rule_right)
                    node.right = node.right.setSampleIndices(X_indices_right)
                    node.right = node.right.setDepth(node.depth + 1)
                    if node.is_changed:
                        node.left.is_changed = node.is_changed
                        node.right.is_changed = node.is_changed
                    stack.append(node.right)
                    stack.append(node.left)
                    node = node.setIndex(idx)
                    idx += 1
                else:
                    child = node.left if X_left.shape[0] > 0 else node.right
                    parent = node.parent
                    if node.branch == "left":
                        parent.left = child
                    else:
                        parent.right = child
                    child.parent = parent
                    child.branch = node.branch
                    child = child.setSample(node.sample)
                    child = child.setSampleRule(node.sample_rule)
                    child = child.setSampleIndices(node.sample_indices)
                    child = child.setDepth(node.depth)
                    if node.is_changed:
                        child.is_changed = node.is_changed
                    stack.append(child)
        # print(parent, parent.left)
        return self.dummy_.left

    def copyTree(self, root):
        root_copy = Node(self.dummy_, branch="left")
        stack = [(root_copy, root)]
        while len(stack) > 0:
            node_copy, node = stack.pop()
            node_copy = node_copy.setDepth(node.depth)
            node_copy = node_copy.setIndex(node.idx)
            node_copy = node_copy.setSample(node.sample)
            node_copy = node_copy.setSampleRule(node.sample_rule)
            if node.is_leaf():
                node_copy = node_copy.setAction(
                    action=node.action,
                    cost=node.cost,
                    active=node.active,
                    objective=node.objective,
                    is_infeasible=node.is_infeasible,
                )
            else:
                node_copy = node_copy.setRule(node.feature)
                right = Node(node_copy, branch="right")
                left = Node(node_copy, branch="left")
                node_copy.right = right
                stack.append((right, node.right))
                node_copy.left = left
                stack.append((left, node.left))
        return root_copy

    def setNodeList(self):
        self.node_list = []
        self.leaf_list = []
        self.internal_list = []
        stack = [self.dummy_.left]
        while len(stack) > 0:
            node = stack.pop()
            self.node_list.append(node)
            if node.is_leaf():
                self.leaf_list.append(node)
            else:
                self.internal_list.append(node)
                stack.append(node.right)
                stack.append(node.left)
        self.leaf_size = len(self.leaf_list)
        self.internal_size = len(self.internal_list)
        self.node_size = self.leaf_size + self.internal_size
        return self

    def is_infeasible(self):
        for leaf in self.leaf_list:
            if leaf.is_infeasible:
                return True
        return False

    def getObjective(self, root):
        g = 0
        comp = 0
        cost = 0
        acts = 0
        stack = [root]
        while len(stack) > 0:
            node = stack.pop()
            if node.is_leaf():
                if node.is_infeasible:
                    return np.inf
                else:
                    g += node.objective / self.N_
                    cost += node.cost / self.N_
                    acts += node.active / self.N_
                    comp += 1
            else:
                stack.append(node.right)
                stack.append(node.left)
        return g + self.lambda_ * comp, cost, 1 - acts, comp

    def fit(
        self,
        X,
        max_change_num=4,
        cost_type="TLPS",
        C=1.0,
        gamma=1.0,
        max_leaf_size=-1,
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
        self.lambda_ = C
        self.leaf_size_bound_ = int((self.gamma_ + self.lambda_) / self.lambda_)
        self.max_leaf_size_ = self.leaf_size_bound_ if max_leaf_size < 1 else max_leaf_size
        self.time_limit_ = time_limit
        self.X_rule_ = self.discretizer_.transform(X)
        if self.lime_approximation_:
            self.lime_coefs_, self.lime_intercepts_ = (
                np.zeros(X.shape),
                np.zeros(self.N_),
            )
            for n, x in enumerate(X):
                self.lime_coefs_[n], self.lime_intercepts_[n] = self.lime_.approximate(x)

        # Initialization
        self.dummy_ = DummyNode()
        self.dummy_.left = self.generateTree()
        root_prev = self.optimizeTree()
        obj_prev, _, _, comp_prev = self.getObjective(root_prev)
        obj_best = obj_prev
        self.n_leaves_ = comp_prev
        self.root_ = self.copyTree(root_prev)

        objs = {
            "Iteration": np.arange(1, self.max_iteration_ + 1),
            "obj": np.zeros(self.max_iteration_),
            "obj_bound": np.zeros(self.max_iteration_),
            "cost": np.zeros(self.max_iteration_),
            "loss": np.zeros(self.max_iteration_),
            "comp": np.zeros(self.max_iteration_),
            "time": np.zeros(self.max_iteration_),
        }
        start = time.perf_counter()

        # Stochastic Local Search
        if verbose:
            logger.info("## Stochastic Local Searching ...")
        for i in range(self.max_iteration_):
            self.dummy_.left = self.copyTree(self.dummy_.left)
            if verbose and (i + 1) % 10 == 0:
                logger.info("### Iteration:", i + 1)
                logger.info("#### Before:")
                self.print_tree(root=self.dummy_.left)
            self = self.setNodeList()
            edit = self.selectEditOperation()
            if edit == "I":
                self.insertNode()
                if verbose and (i + 1) % 10 == 0:
                    logger.info("#### After 'Insert'")
            elif edit == "D":
                self.deleteNode()
                if verbose and (i + 1) % 10 == 0:
                    logger.info("#### After 'Delete'")
            elif edit == "R":
                self.replaceNode()
                if verbose and (i + 1) % 10 == 0:
                    logger.info("#### After 'Replace'")
            root_next = self.optimizeTree()
            obj_next, cost_next, loss_next, comp_next = self.getObjective(root_next)
            if verbose and (i + 1) % 10 == 0:
                self.print_tree(root=root_next)
                logger.info("##### Score:")
                # logger.info("- Objective: {} => {}".format(obj_prev, obj_next))
                # logger.info("- Update:", self.isUpdate(i, obj_prev, obj_next))
                # logger.info("- Best:", obj_next < obj_best)
                # logger.info("- Time:", time.perf_counter() - start)
            if self.isUpdate(i, obj_prev, obj_next):
                root_prev = root_next
                obj_prev = obj_next
                self.dummy_.left = root_next
            else:
                self.dummy_.left = root_prev
            if obj_next < obj_best:
                obj_best = obj_next
                self.root_ = self.copyTree(root_next)
                self.n_leaves_ = comp_next
            objs["obj"][i] = obj_next
            objs["obj_bound"][i] = obj_best
            objs["cost"][i] = cost_next
            objs["loss"][i] = loss_next
            objs["comp"][i] = comp_next
            objs["time"][i] = time.perf_counter() - start

        # if(verbose): print('# Best (obj. = {}): '.format(obj_best)); self.print_tree();
        self.objs_ = objs
        self.time_ = time.perf_counter() - start
        return self

    def selectEditOperation(self):
        if self.leaf_size >= self.leaf_size_bound_ or self.leaf_size >= self.max_leaf_size_:
            p = np.array([0, 1, 1])
        elif self.internal_size < 1:
            p = np.array([1, 0, 0])
        else:
            p = np.array(
                [
                    1 - (self.leaf_size / self.leaf_size_bound_),
                    self.leaf_size / self.leaf_size_bound_,
                    1 / 2,
                ]
            )
        return np.random.choice(["I", "D", "R"], p=p / p.sum())

    def selectNode(self, node="all", priority="uniform"):
        if node == "leaf":
            nodes = self.leaf_list
            size = self.leaf_size
        elif node == "internal":
            nodes = self.internal_list
            size = self.internal_size
        elif node == "all":
            nodes = self.node_list
            size = self.node_size
        if priority == "uniform":
            p = np.ones(size)
        elif priority == "objective":
            p = np.array([n.getObjective() for n in nodes])
        elif priority == "active":
            p = np.array([n.getLoss() for n in nodes])
        elif priority == "dense":
            p = np.array([n.n_sample() for n in nodes])
        elif priority == "sparse":
            p = np.array([self.N_ - n.n_sample() for n in nodes])
        if p.sum() == 0:
            p = np.ones(size)
        return np.random.choice(nodes, p=p / p.sum())

    def insertNode(self):
        # node = self.selectNode(priority='dense')
        node = self.selectNode(priority="active")
        node_ins = Node(node.parent, branch=node.branch)
        r = self.selectRule(node=node)
        node_ins = node_ins.setRule(r)
        leaf_ins = Node(node_ins, branch="right")
        node_ins.right = leaf_ins
        node_ins.left = node
        node_ins.is_changed = True
        if node.branch == "left":
            node.parent.left = node_ins
        else:
            node.parent.right = node_ins
        node.parent = node_ins
        node.branch = "left"
        return

    def deleteNode(self):
        node = self.selectNode(node="leaf", priority="sparse")
        parent = node.parent
        grandparent = parent.parent
        subtree = parent.left if node.branch == "right" else parent.right
        if parent.branch == "left":
            grandparent.left = subtree
        else:
            grandparent.right = subtree
        subtree.parent = grandparent
        subtree.branch = parent.branch
        subtree.is_changed = True
        del node
        return

    def switchNode(self):
        while True:
            node1 = self.selectNode(node="internal")
            node2 = self.selectNode(node="internal")
            if node1.feature != node2.feature:
                break
        r1 = node1.feature
        r2 = node2.feature
        node1 = node1.setRule(r2)
        node2 = node2.setRule(r1)
        node1.is_changed = True
        node2.is_changed = True
        return

    def replaceNode(self):
        node = self.selectNode(node="internal")
        r = self.selectRule(node=node)
        node = node.setRule(r)
        node.is_changed = True
        return

    def selectRule(self, node=None):
        if node is None:
            return np.random.choice(range(self.R_), p=self.rule_probability_)
        else:
            R_candidates = [
                r
                for r in range(self.R_)
                if node.sample_rule[:, r].mean() > 0 and node.sample_rule[:, r].mean() < 1
            ]
            if len(R_candidates) == 0:
                return np.random.choice(range(self.R_), p=self.rule_probability_)
            else:
                return np.random.choice(R_candidates)

    def isUpdate(self, i, obj_prev, obj_next, C=0.01):
        return np.random.random() <= np.exp(
            float((obj_prev - obj_next) / (C ** (i / self.max_iteration_)))
        )

    def feasify(self, a, x):
        for d in [d for d in range(self.D_) if self.feature_types_[d] == "B"]:
            x_d = x[d] + a[d]
            if x_d not in [0, 1]:
                a[d] = 0
        for G in self.feature_categories_:
            x_G = x[G] + a[G]
            if x_G.sum() != 1:
                a[G] = 0
        return a

    def predict(self, X):
        X_rule = self.discretizer_.transform(X)
        A = [self.root_.predict(x) for x in X_rule]
        return np.array([self.feasify(a, x) for a, x in zip(A, X)])

    def cost(self, X, cost_type="TLPS"):
        A = self.predict(X)
        return np.array(
            [self.cost_.compute(x, a, cost_type=cost_type) for x, a in zip(X, A)]
        ).mean()

    def loss(self, X, target=0):
        A = self.predict(X)
        return (self.mdl_.predict(X + A) != target).mean()

    def print_tree(self, root=None):
        def rec(node, depth):
            if node.is_leaf():
                if node.is_infeasible:
                    print(
                        "\t" * depth
                        + "* Action ({}/{} = {:.1%}): No Feasible Action".format(
                            node.active, node.sample.shape[0], node.getRatio()
                        )
                    )
                else:
                    print(
                        "\t" * depth
                        + "*{}Action [{}: {} -> {}] ({}/{} = {:.1%} / MeanCost = {:.3}):".format(
                            " (+) " if node.is_changed else " ",
                            self.target_name_,
                            self.target_labels_[1],
                            self.target_labels_[0],
                            node.active,
                            node.sample.shape[0],
                            node.getRatio(),
                            node.cost / node.sample.shape[0],
                        )
                    )
                    for i, d in enumerate(np.where(abs(node.action) > 1e-8)[0]):
                        g = self.feature_categories_inv_[d]
                        if g == -1:
                            if self.feature_types_[d] == "C":
                                print(
                                    "\t" * (depth + 1)
                                    + "* {}: {:+.4f}".format(self.feature_names_[d], node.action[d])
                                )
                            elif self.feature_types_[d] == "B":
                                if node.action[d] == -1:
                                    print(
                                        "\t" * (depth + 1)
                                        + "* {}: True -> False".format(
                                            self.feature_names_[d],
                                        )
                                    )
                                else:
                                    print(
                                        "\t" * (depth + 1)
                                        + "* {}: False -> True".format(
                                            self.feature_names_[d],
                                        )
                                    )
                            else:
                                print(
                                    "\t" * (depth + 1)
                                    + "* {}: {:+}".format(
                                        self.feature_names_[d],
                                        node.action[d].astype(int),
                                    )
                                )
                        else:
                            if node.action[d] == -1:
                                continue
                            cat_name, nxt = self.feature_names_[d].split(":")
                            cat = self.feature_categories_[g]
                            prv = self.feature_names_[
                                cat[np.where(node.action[cat] == -1)[0][0]]
                            ].split(":")[1]
                            print(
                                "\t" * (depth + 1) + '* {}: "{}" -> "{}"'.format(cat_name, prv, nxt)
                            )
            else:
                print(
                    "\t" * depth
                    + "-{}If {}:".format(
                        " (+) " if node.is_changed else " ",
                        self.rule_names_[node.feature],
                    )
                )
                rec(node.right, depth + 1)
                print("\t" * depth + "- Else:")
                rec(node.left, depth + 1)

        if root is None:
            root = self.root_
        rec(root, 0)
        print()

    def get_counterfactuals(self, X, return_costs=False):
        """
        Get counterfactual examples for input instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input instances to generate counterfactuals for
        return_costs : bool, default=False
            Whether to return the costs associated with each counterfactual

        Returns
        -------
        counterfactuals : array-like of shape (n_samples, n_features)
            Counterfactual examples
        costs : array-like of shape (n_samples,), optional
            Costs associated with each counterfactual if return_costs=True
        """
        actions = self.predict(X)
        counterfactuals = X + actions

        if return_costs:
            # Compute costs for each counterfactual
            costs = np.array(
                [self.cost_.compute(x, a, cost_type=self.cost_type_) for x, a in zip(X, actions)]
            )
            return counterfactuals, costs

        return counterfactuals


def _check(
    dataset="h",
    model="L",
    params=(0.01, 0.75),
    max_iteration=1000,
    lime_approximation=False,
    verbose=False,
):
    np.random.seed(0)
    LAMBDA, GAMMA = params
    MAX_ITERATION = max_iteration
    from utils import DatasetHelper

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)

    print("# Learing Hierarchical Actionable Recourse Summary")
    if model == "L":
        print("* Classifier: LogisticRegression")
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced")
        print("\t* C: {}".format(mdl.C))
        print("\t* penalty: {}".format(mdl.penalty))
    elif model == "F":
        print("* Classifier: RandomForest")
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, class_weight="balanced")
        print("\t* n_estimators: {}".format(mdl.n_estimators))
        print("\t* max_leaf_nodes: {}".format(mdl.max_leaf_nodes))
    elif model == "M":
        print("* Classifier: MultiLayerPerceptron")
        from sklearn.neural_network import MLPClassifier

        mdl = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, activation="relu", alpha=0.0001)
        print("\t* hidden_layer_size: {}".format(mdl.hidden_layer_sizes[0]))
        print("\t* activation: {}".format(mdl.activation))
    elif model == "X":
        print("* Classifier: LightGBM")
        from lightgbm import LGBMClassifier

        mdl = LGBMClassifier(n_estimators=100, num_leaves=16)
        print("\t* n_estimators: {}".format(mdl.n_estimators))
        print("\t* num_leaves: {}".format(mdl.num_leaves))
    elif model == "T":
        print("* Classifier: TabNet")
        from utils import MyTabNetClassifier

        mdl = MyTabNetClassifier(D.feature_types, verbose=int(verbose))

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    print("* Dataset:", D.dataset_fullname)
    for d in range(D.n_features):
        print(
            "\t* x_{:<2}: {} ({}{})".format(
                d + 1,
                D.feature_names[d],
                D.feature_types[d],
                ":" + D.feature_constraints[d] if D.feature_constraints[d] != "" else "",
            )
        )

    cet = CounterfactualExplanationTree(
        mdl,
        X_tr,
        y_tr,
        max_iteration=MAX_ITERATION,
        lime_approximation=lime_approximation,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )
    print("* Parameters:")
    print("\t* lambda: {}".format(LAMBDA))
    print("\t* gamma: {}".format(GAMMA))
    print("\t* max_iteration: {}".format(MAX_ITERATION))
    print("\t* leaf size bound:", int((GAMMA + LAMBDA) / LAMBDA))
    print("\t* LIME approximation:", lime_approximation)
    print()

    X = X_ts[mdl.predict(X_ts) == 1]
    cet = cet.fit(
        X,
        max_change_num=3,
        cost_type="MPS",
        C=LAMBDA,
        gamma=GAMMA,
        time_limit=60,
        verbose=verbose,
    )
    print("## Learned CET")
    cet.print_tree()

    print("### Score:")
    cost = cet.cost(X, cost_type="MPS")
    loss = cet.loss(X)
    print("- Train (N={}):".format(X.shape[0]))
    print("\t- cost: {}".format(cost))
    print("\t- loss: {}".format(loss))
    print("\t- g(a|x): {}".format(cost + GAMMA * loss))
    print("\t- obj.: {}".format(cet.objs_["obj"].min()))

    X = X_ts[mdl.predict(X_ts) == 1]
    cost = cet.cost(X, cost_type="MPS")
    loss = cet.loss(X)
    print("- Test (N={}):".format(X.shape[0]))
    print("\t- cost: {}".format(cost))
    print("\t- loss: {}".format(loss))
    print("\t- g(a|x): {}".format(cost + GAMMA * loss))
    print()


if __name__ == "__main__":
    MAX_ITERATION = 50
    _check(
        dataset="h",
        model="X",
        params=(0.02, 1.0),
        max_iteration=MAX_ITERATION,
        lime_approximation=False,
        verbose=True,
    )
