import numpy as np
import time
import pulp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from .utils import (
    flatten,
    LimeEstimator,
    Action,
    ActionCandidates,
    ForestActionCandidates,
)


class ActionExtractor:
    def __init__(
        self,
        mdl,
        X,
        Y=[],
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        max_candidates=100,
        tol=1e-6,
        target_name="Output",
        target_labels=["Good", "Bad"],
        lime_approximation=False,
        n_samples=5000,
        alpha=1.0,
    ):
        self.mdl_ = mdl
        self.lime_approximation_ = lime_approximation
        if isinstance(mdl, LogisticRegression):
            self.coef_ = mdl.coef_[0]
            self.intercept_ = mdl.intercept_[0]
            self.AC_ = ActionCandidates(
                X,
                Y=Y,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_categories=feature_categories,
                feature_constraints=feature_constraints,
                max_candidates=max_candidates,
                tol=tol,
            )
            self.T_ = len(mdl.coef_[0])
            self.lime_approximation_ = False
        elif lime_approximation:
            self.lime_ = LimeEstimator(
                mdl,
                X,
                n_samples=n_samples,
                feature_types=feature_types,
                feature_categories=feature_categories,
                alpha=alpha,
            )
            self.AC_ = ActionCandidates(
                X,
                Y=Y,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_categories=feature_categories,
                feature_constraints=feature_constraints,
                max_candidates=max_candidates,
                tol=tol,
            )
            self.T_ = X.shape[1]
        elif isinstance(mdl, RandomForestClassifier):
            self.T_ = mdl.n_estimators
            self.coef_ = np.ones(self.T_) / self.T_
            self.intercept_ = -1 * 0.5
            self.AC_ = ForestActionCandidates(
                X,
                mdl,
                Y=Y,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_categories=feature_categories,
                feature_constraints=feature_constraints,
                max_candidates=max_candidates,
                tol=tol,
            )
            self.L_ = self.AC_.L_
            self.H_ = self.AC_.H_
        elif isinstance(mdl, MLPClassifier):
            self.hidden_coef_ = mdl.coefs_[0]
            self.coef_ = mdl.coefs_[1]
            self.hidden_intercept_ = mdl.intercepts_[0]
            self.intercept_ = mdl.intercepts_[1][0]
            self.AC_ = ActionCandidates(
                X,
                Y=Y,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_categories=feature_categories,
                feature_constraints=feature_constraints,
                max_candidates=max_candidates,
                tol=tol,
            )
            self.T_ = mdl.intercepts_[0].shape[0]
        else:
            self.lime_approximation_ = True
            self.lime_ = LimeEstimator(
                mdl,
                X,
                n_samples=n_samples,
                feature_types=feature_types,
                feature_categories=feature_categories,
                alpha=alpha,
            )
            self.AC_ = ActionCandidates(
                X,
                Y=Y,
                feature_names=feature_names,
                feature_types=feature_types,
                feature_categories=feature_categories,
                feature_constraints=feature_constraints,
                max_candidates=max_candidates,
                tol=tol,
            )
            self.T_ = X.shape[1]

        self.D_ = X.shape[1]
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

    def __getProblem(
        self,
        X,
        y,
        A=None,
        C=None,
        I=None,  # noqa: E741
        lime_coefs=None,
        lime_intercepts=None,
        immutables=[],
        max_change_num=4,
        cost_type="TLPS",
        tradeoff_parameter=1.0,
        hinge_relax=False,
    ):
        self.N_ = X.shape[0]
        self.K_ = min(max_change_num, self.D_)
        self.gamma_ = tradeoff_parameter
        if A is None and C is None:
            s = time.perf_counter()
            (
                A,
                C,
            ) = self.AC_.generateActions(X, y, cost_type=cost_type, multi=True)

        s = time.perf_counter()
        prob = pulp.LpProblem()
        self.lb_ = [np.min(A_d) for A_d in A]
        self.ub_ = [np.max(A_d) for A_d in A]
        non_zeros = []
        for d in range(self.D_):
            non_zeros_d = [1] * len(A[d])
            for i in range(len(A[d])):
                if d in flatten(self.feature_categories_):
                    if A[d][i] <= 0:
                        non_zeros_d[i] = 0
                elif A[d][i] == 0:
                    non_zeros_d[i] = 0
            non_zeros.append(non_zeros_d)

        # variables: action
        self.variables_ = {}
        act = [
            pulp.LpVariable(
                "act_{:04d}".format(d),
                cat="Continuous",
                lowBound=self.lb_[d],
                upBound=self.ub_[d],
            )
            for d in range(self.D_)
        ]
        self.variables_["act"] = act
        pi = [
            [
                pulp.LpVariable("pi_{:04d}_{:04d}".format(d, i), cat="Binary")
                for i in range(len(A[d]))
            ]
            for d in range(self.D_)
        ]
        self.variables_["pi"] = pi
        cost = [
            pulp.LpVariable("cost_{:04d}".format(n), cat="Continuous", lowBound=0)
            for n in range(self.N_)
        ]
        self.variables_["cost"] = cost
        if hinge_relax:
            omega = [
                pulp.LpVariable("omg_{:04d}".format(n), cat="Continuous", lowBound=0)
                for n in range(self.N_)
            ]
            self.variables_["omega"] = omega
        else:
            omega = [
                pulp.LpVariable("omg_{:04d}".format(n), cat="Binary")
                for n in range(self.N_)
            ]
            self.variables_["omega"] = omega

        # variables and constants: base learner
        if isinstance(self.mdl_, LogisticRegression) or self.lime_approximation_:
            xi = [
                [
                    pulp.LpVariable(
                        "xi_{:04d}_{:04d}".format(n, d),
                        cat="Continuous",
                        lowBound=X[n, d] + self.lb_[d],
                        upBound=X[n, d] + self.ub_[d],
                    )
                    for d in range(self.D_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["xi"] = xi
            y_dec = self.mdl_.predict_proba(X)
        elif isinstance(self.mdl_, RandomForestClassifier):
            xi = [
                [
                    pulp.LpVariable(
                        "xi_{:04d}_{:04d}".format(n, t),
                        cat="Continuous",
                        lowBound=0,
                        upBound=1,
                    )
                    for t in range(self.T_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["xi"] = xi
            phi = [
                [
                    [
                        pulp.LpVariable(
                            "phi_{:04d}_{:04d}_{:04d}".format(n, t, l), cat="Binary"
                        )
                        for l in range(self.L_[t])  # noqa: E741
                    ]
                    for t in range(self.T_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["phi"] = phi
            if I is None:
                I = self.AC_.I_  # noqa: E741
        elif isinstance(self.mdl_, MLPClassifier):
            xi = [
                [
                    pulp.LpVariable(
                        "xi_{:04d}_{:04d}".format(n, t), cat="Continuous", lowBound=0
                    )
                    for t in range(self.T_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["xi"] = xi
            bxi = [
                [
                    pulp.LpVariable(
                        "bxi_{:04d}_{:04d}".format(n, t), cat="Continuous", lowBound=0
                    )
                    for t in range(self.T_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["bxi"] = bxi
            nu = [
                [
                    pulp.LpVariable("nu_{:04d}_{:04d}".format(n, t), cat="Binary")
                    for t in range(self.T_)
                ]
                for n in range(self.N_)
            ]
            self.variables_["nu"] = nu
            M_bar, M = np.zeros(self.T_), np.zeros(self.T_)
            for t, w in enumerate(self.hidden_coef_.T):
                M_bar[t] += np.sum(
                    [
                        min(w[d] * self.ub_[d], w[d] * self.lb_[d])
                        for d in range(self.D_)
                    ]
                )
                M[t] += np.sum(
                    [
                        max(w[d] * self.ub_[d], w[d] * self.lb_[d])
                        for d in range(self.D_)
                    ]
                )

        # objective function: sum_{x \in X} C(a | x) + gamma * omega_x
        prob += pulp.lpSum(cost) + tradeoff_parameter * pulp.lpSum(omega)
        prob += pulp.lpSum(cost) + tradeoff_parameter * pulp.lpSum(omega) >= 0

        # constraint: sum_{i} pi_{d,i} == 1
        for d in range(self.D_):
            prob.addConstraint(
                pulp.lpSum(pi[d]) == 1, name="C_basic_pi_{:04d}".format(d)
            )

        # constraint: a_d = sum_{i} a_{d,i} pi_{d,i}
        for d in range(self.D_):
            prob.addConstraint(
                act[d] - pulp.lpDot(A[d], pi[d]) == 0,
                name="C_basic_act_{:04d}".format(d),
            )

        # constraint: sum_{d} sum_{i} pi_{d,i} <= K
        if self.K_ >= 1:
            prob.addConstraint(
                pulp.lpDot(flatten(non_zeros), flatten(pi)) <= self.K_,
                name="C_basic_sparsity",
            )

        # constraint: sum_{i} pi_{d,i} == 0 for d in immutables
        if len(immutables) > 0:
            for d in immutables:
                prob.addConstraint(
                    pulp.lpDot(non_zeros[d], pi[d]) == 0,
                    name="C_basic_immutable_{:04d}".format(d),
                )

        # constraint: sum_{d in G} a_d = 0
        for i, G in enumerate(self.feature_categories_):
            prob.addConstraint(
                pulp.lpSum([act[d] for d in G]) == 0,
                name="C_basic_category_{:04d}".format(i),
            )

        self.constraints_ = {}
        for n in range(self.N_):
            self.constraints_[n] = []
            # constraint: C(a | x) = sum_{d} sum_{i} c_{d,i} pi_{d,i}
            if cost_type == "MPS":
                for d in range(self.D_):
                    if (
                        d in flatten(self.feature_categories_) and np.min(A[d]) < 0
                    ) or self.feature_constraints_[d] == "FIX":
                        continue
                    prob.addConstraint(
                        cost[n] - pulp.lpDot(C[n][d], pi[d]) >= 0,
                        name="C_{:04d}_cost_{:04d}".format(n, d),
                    )
                    self.constraints_[n] += ["C_{:04d}_cost_{:04d}".format(n, d)]
            else:
                prob.addConstraint(
                    cost[n] - pulp.lpDot(flatten(C[n]), flatten(pi)) == 0,
                    name="C_{:04d}_cost".format(n),
                )
                self.constraints_[n] += ["C_{:04d}_cost".format(n)]

            # constraints: omega = l(y_target, h(x+a))
            if hinge_relax:
                # constraint: omega = l_hinge(y_target, h(x+a))
                y_target = 1 if y == 0 else -1
                # constraint: omega >= 1 - y_target * (sum_{d} w_t xi_{n,t} + b)
                prob.addConstraint(
                    omega[n] + y_target * pulp.lpDot(self.coef_, xi[n])
                    >= 1 - y_target * self.intercept_,
                    name="C_{:04d}_loss".format(n),
                )
                self.constraints_[n] += ["C_{:04d}_loss".format(n)]
            else:
                # constraint: omega = I[h(x+a)!=h(x)]
                if self.lime_approximation_:
                    M_min = -1e4
                    M_max = 1e4
                    if lime_coefs is None and lime_intercepts is None:
                        coef, intercept = self.lime_.approximate(X[n])
                    else:
                        coef, intercept = lime_coefs[n], lime_intercepts[n]
                    # constraint: sum_{d} w_t xi_{n,t} + b >= M_min * omega
                    prob.addConstraint(
                        pulp.lpDot(coef, xi[n]) - M_min * omega[n] >= -intercept + 1e-8,
                        name="C_{:04d}_loss_ge".format(n),
                    )
                    self.constraints_[n] += ["C_{:04d}_loss_ge".format(n)]
                    # constraint: sum_{d} w_t xi_{n,t} + b <= M_max * (1-omega)
                    prob.addConstraint(
                        pulp.lpDot(coef, xi[n]) + M_max * omega[n] <= M_max - intercept,
                        name="C_{:04d}_loss_le".format(n),
                    )
                    self.constraints_[n] += ["C_{:04d}_loss_le".format(n)]
                else:
                    if isinstance(self.mdl_, RandomForestClassifier):
                        M_min = -1.0
                        M_max = 1.0
                    else:
                        M_min = -1e4
                        M_max = 1e4
                    if y == 0:
                        # constraint: sum_{d} w_t xi_{n,t} + b >= M_min * omega
                        prob.addConstraint(
                            pulp.lpDot(self.coef_, xi[n]) - M_min * omega[n]
                            >= -self.intercept_ + 1e-8,
                            name="C_{:04d}_loss_ge".format(n),
                        )
                        self.constraints_[n] += ["C_{:04d}_loss_ge".format(n)]
                        # constraint: sum_{d} w_t xi_{n,t} + b <= M_max * (1-omega)
                        prob.addConstraint(
                            pulp.lpDot(self.coef_, xi[n]) + M_max * omega[n]
                            <= M_max - self.intercept_,
                            name="C_{:04d}_loss_le".format(n),
                        )
                        self.constraints_[n] += ["C_{:04d}_loss_le".format(n)]
                    else:
                        # constraint: sum_{d} w_t xi_{n,t} + b <= M_max * omega
                        prob.addConstraint(
                            pulp.lpDot(self.coef_, xi[n]) - M_max * omega[n]
                            <= -self.intercept_ - 1e-8,
                            name="C_{:04d}_loss_le".format(n),
                        )
                        self.constraints_[n] += ["C_{:04d}_loss_le".format(n)]
                        # constraint: sum_{d} w_t xi_{n,t} + b >= M_min * (1-omega)
                        prob.addConstraint(
                            pulp.lpDot(self.coef_, xi[n]) + M_min * omega[n]
                            >= M_min - self.intercept_,
                            name="C_{:04d}_loss_ge".format(n),
                        )
                        self.constraints_[n] += ["C_{:04d}_loss_ge".format(n)]

            # constraint (Linear model): xi_d = x_d + a_d
            if isinstance(self.mdl_, LogisticRegression) or self.lime_approximation_:
                # constraint: xi_d = x_d + a_d
                for d in range(self.D_):
                    prob.addConstraint(
                        xi[n][d] - act[d] == X[n, d],
                        name="C_{:04d}_linear_{:04d}".format(n, d),
                    )
                    self.constraints_[n] += ["C_{:04d}_linear_{:04d}".format(n, d)]

            # constraints (Tree Ensemble):
            elif isinstance(self.mdl_, RandomForestClassifier):
                for t in range(self.T_):
                    # constraint: sum_{l} phi_{t,l} = 1
                    prob.addConstraint(
                        pulp.lpSum(phi[n][t]) == 1,
                        name="C_{:04d}_forest_leaf_{:04d}".format(n, t),
                    )
                    self.constraints_[n] += ["C_{:04d}_forest_leaf_{:04d}".format(n, t)]
                    # constraint: xi_t = sum_{l} h_{t,l} phi_{t,l}
                    prob.addConstraint(
                        xi[n][t] - pulp.lpDot(self.H_[t], phi[n][t]) == 0,
                        name="C_{:04d}_forest_{:04d}".format(n, t),
                    )
                    self.constraints_[n] += ["C_{:04d}_forest_{:04d}".format(n, t)]
                    # constraint: D * phi_{t,l} <= sum_{d} sum_{i in I_{t,l,d}} pi_{d,i}
                    for l in range(self.L_[t]):  # noqa: E741
                        p = self.AC_.ancestors_[t][l]
                        prob.addConstraint(
                            len(p) * phi[n][t][l]
                            - pulp.lpDot(
                                flatten([I[n][t][l][d] for d in p]),
                                flatten([pi[d] for d in p]),
                            )
                            <= 0,
                            name="C_{:04d}_forest_decision_{:04d}_{:04d}".format(
                                n, t, l
                            ),
                        )
                        self.constraints_[n] += [
                            "C_{:04d}_forest_decision_{:04d}_{:04d}".format(n, t, l)
                        ]

            # constraints (Multi-Layer Perceptoron):
            elif isinstance(self.mdl_, MLPClassifier):
                M_bar_n = -1 * (
                    X[n].dot(self.hidden_coef_) + self.hidden_intercept_ + M_bar
                )
                M_n = X[n].dot(self.hidden_coef_) + self.hidden_intercept_ + M
                M_bar_n[M_bar_n < 0] = 0.0
                M_n[M_n < 0] = 0.0
                M_bar_n[M_bar_n > 0] += self.tol_
                M_n[M_n > 0] += self.tol_

                for t in range(self.T_):
                    ## constraint: xi_t <= M_t nu_t
                    ## constraint: bxi_t <= M_bar_t (1-nu_t)
                    prob.addConstraint(
                        xi[n][t] - M_n[t] * nu[n][t] <= 0,
                        name="C_{:04d}_mlp_pos_{:04d}".format(n, t),
                    )
                    self.constraints_[n] += ["C_{:04d}_mlp_pos_{:04d}".format(n, t)]
                    prob.addConstraint(
                        bxi[n][t] + M_bar_n[t] * nu[n][t] <= M_bar_n[t],
                        name="C_{:04d}_mlp_neg_{:04d}".format(n, t),
                    )
                    self.constraints_[n] += ["C_{:04d}_mlp_neg_{:04d}".format(n, t)]

                    ## constraint: xi_t = bxi_t + sum_{d} w_{t,d} (x_d + a_d) + b_t
                    prob.addConstraint(
                        xi[n][t] - bxi[n][t] - pulp.lpDot(self.hidden_coef_.T[t], act)
                        == X[n].dot(self.hidden_coef_.T[t]) + self.hidden_intercept_[t],
                        name="C_{:04d}_mlp_{:04d}".format(n, t),
                    )
                    self.constraints_[n] += ["C_{:04d}_mlp_{:04d}".format(n, t)]

        self.actions_ = A
        # t = time.perf_counter()-s; print('Build Model: {}[s]'.format(t))
        return prob

    def extract(
        self,
        X,
        max_change_num=4,
        cost_type="TLPS",
        tradeoff_parameter=1.0,
        hinge_relax=False,
        lime_coefs=None,
        lime_intercepts=None,
        immutables=[],
        solver="cplex",
        time_limit=180,
        log_stream=False,
        log_name="",
        init_sols={},
        verbose=False,
    ):
        if X.shape == (self.D_,):
            X = X.reshape(1, -1)

        y = self.mdl_.predict(X[0].reshape(1, -1)).item()
        target_label = int(1 - y)
        prob = self.__getProblem(
            X,
            y,
            lime_coefs=lime_coefs,
            lime_intercepts=lime_intercepts,
            immutables=immutables,
            max_change_num=max_change_num,
            cost_type=cost_type,
            tradeoff_parameter=tradeoff_parameter,
            hinge_relax=hinge_relax,
        )

        if len(log_name) != 0:
            prob.writeLP(log_name + ".lp")
        s = time.perf_counter()
        prob.solve(solver=pulp.CPLEX_PY(msg=log_stream, timeLimit=time_limit))
        t = time.perf_counter() - s

        if prob.status != 1:
            prob.writeLP("infeasible.lp")
            prob.solve(
                solver=pulp.CPLEX_PY(msg=True, options=["set output clonelog -1"])
            )
            return {
                "feasible": False,
                "sample": X.shape[0],
                "action": np.zeros(X.shape[1]),
                "cost": -1 * np.ones(X.shape[0]),
                "active": np.zeros(X.shape[0]),
                "instance": X,
                "objective": -1,
                "time": t,
            }
        else:
            # save initial values
            self.init_sols_ = {}
            self.init_sols_["act"] = [a.value() for a in self.variables_["act"]]
            self.init_sols_["pi"] = []
            for pi_d in self.variables_["pi"]:
                self.init_sols_["pi"].append([round(p.value()) for p in pi_d])

            a = np.array(
                [
                    np.sum(
                        [
                            self.actions_[d][i]
                            * round(self.variables_["pi"][d][i].value())
                            for i in range(len(self.actions_[d]))
                        ]
                    )
                    for d in range(X.shape[1])
                ]
            )
            return {
                "feasible": True,
                "sample": X.shape[0],
                "action": a,
                "cost": np.array([c.value() for c in self.variables_["cost"]]),
                "loss": np.array([o.value() for o in self.variables_["omega"]]),
                "active": self.mdl_.predict(X + a) == target_label,
                "instance": X,
                # 'objective': prob.objective.value(),
                "objective": np.array(
                    [c.value() for c in self.variables_["cost"]]
                ).sum()
                + tradeoff_parameter * (self.mdl_.predict(X + a) != target_label).sum(),
                "time": t,
            }

    def getActionObject(self, action_dict):
        X = action_dict["instance"]
        return [
            Action(
                X[n],
                action_dict["action"],
                scores={
                    "cost": action_dict["cost"][n],
                    "loss": action_dict["loss"][n],
                    "active": action_dict["active"][n],
                    "probability": self.mdl_.predict_proba(
                        (X[n] + action_dict["action"]).reshape(1, -1)
                    )[0],
                },
                target_name=self.target_name_,
                target_labels=self.target_labels_,
                label_before=int(self.mdl_.predict(X[n].reshape(1, -1))[0]),
                label_after=int(
                    self.mdl_.predict((X[n] + action_dict["action"]).reshape(1, -1))[0]
                ),
                feature_names=self.feature_names_,
                feature_types=self.feature_types_,
                feature_categories=self.feature_categories_,
            )
            for n in range(X.shape[0])
        ]


def _check_ce(N=1, dataset="h", model="L", hinge_relax=False):
    from utils import DatasetHelper

    np.random.seed(0)

    if model == "L":
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    elif model == "F":
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=16, class_weight="balanced"
        )
    elif model == "M":
        from sklearn.neural_network import MLPClassifier

        mdl = MLPClassifier(
            hidden_layer_sizes=(200,), max_iter=500, activation="relu", alpha=0.0001
        )

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    denied = X_ts[mdl.predict(X_ts) == 1]

    ce = ActionExtractor(
        mdl,
        X_tr,
        Y=y_tr,
        max_candidates=50,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )

    action = ce.extract(
        denied[:N],
        max_change_num=4,
        cost_type="MPS",
        tradeoff_parameter=0.75,
        hinge_relax=hinge_relax,
    )
    if action["feasible"]:
        for n, r in enumerate(ce.getActionObject(action)):
            print("## {}-th instance:".format(n + 1))
            print(r)
        print("## Summary:")
        print("* Perturbation:")
        for d, a_d in enumerate(action["action"]):
            if abs(a_d) > 1e-8:
                print(
                    "\t* {}: {:+.4f}".format(D.feature_names[d], a_d)
                    if D.feature_types[d] == "C"
                    else "\t* {}: {:+}".format(D.feature_names[d], a_d)
                )
        print("* Score:")
        print("\t* Cost:", action["cost"].mean())
        print("\t* Loss:", (1 - action["active"]).mean())
        print("\t* gamma:", ce.gamma_)
        print("\t* Objective:", action["objective"] / N)
        print("\t* Time[s]:", action["time"])


def _check_sens(N=1, dataset="h"):
    from sklearn.linear_model import LogisticRegression
    from utils import DatasetHelper
    import pandas as pd
    from matplotlib import pyplot as plt

    np.random.seed(0)

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    mdl = mdl.fit(X_tr, y_tr)
    denied = X_ts[mdl.predict(X_ts) == 1]
    ce = ActionExtractor(
        mdl,
        X_tr,
        Y=y_tr,
        max_candidates=50,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )

    res = {"gamma": np.arange(0.1, 1.1, 0.1), "cost": [], "loss": []}
    for gamma in res["gamma"]:
        action = ce.extract(
            denied[:N], max_change_num=4, cost_type="MPS", tradeoff_parameter=gamma
        )
        c = action["cost"].mean()
        a = (1 - action["active"]).mean()
        res["cost"].append(c)
        res["loss"].append(a)
        print(
            "\t* Obj. = {:.3} + {:.3} * {:.3} = {:.3}".format(
                c, gamma, a, action["objective"] / N
            )
        )

    if False:
        df = pd.DataFrame(res)
        df.to_csv("linear_sens_{}.csv".format(D.dataset_name), index=False)
        plt.plot(res["gamma"], res["cost"], label="Mean Cost")
        plt.plot(res["gamma"], res["loss"], label="Mean Loss")
        plt.xlabel("gamma")
        plt.legend()
        plt.title("Sensitivity of Trade-off Parameter ({})".format(D.dataset_fullname))
        plt.savefig("sens_{}.png".format(D.dataset_name))


def _check_lime(N=1, dataset="h", model="L", compare=False):
    from utils import DatasetHelper

    np.random.seed(0)

    if model == "L":
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", class_weight="balanced"
        )
    elif model == "F":
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=16, class_weight="balanced"
        )
    elif model == "M":
        from sklearn.neural_network import MLPClassifier

        mdl = MLPClassifier(
            hidden_layer_sizes=(200,), max_iter=500, activation="relu", alpha=0.0001
        )

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    denied = X_ts[mdl.predict(X_ts) == 1]

    if compare:
        print("# Without LIME Approximation")
        ce = ActionExtractor(
            mdl,
            X_tr,
            Y=y_tr,
            max_candidates=50,
            feature_names=D.feature_names,
            feature_types=D.feature_types,
            feature_categories=D.feature_categories,
            feature_constraints=D.feature_constraints,
            target_name=D.target_name,
            target_labels=D.target_labels,
            lime_approximation=False,
        )
        action = ce.extract(
            denied[:N], max_change_num=4, cost_type="MPS", tradeoff_parameter=0.75
        )
        if action["feasible"]:
            print("## Summary:")
            print("* Perturbation:")
            for d, a_d in enumerate(action["action"]):
                if abs(a_d) > 1e-8:
                    print(
                        "\t* {}: {:+.4f}".format(D.feature_names[d], a_d)
                        if D.feature_types[d] == "C"
                        else "\t* {}: {:+}".format(D.feature_names[d], a_d)
                    )
            print("* Score:")
            print("\t* Cost:", action["cost"].mean())
            print("\t* Loss:", (1 - action["active"]).mean())
            print("\t* gamma:", ce.gamma_)
            print("\t* Objective:", action["objective"] / N)
            print("\t* Time[s]:", action["time"])
        print()

    print("# With LIME Approximation")
    ce = ActionExtractor(
        mdl,
        X_tr,
        Y=y_tr,
        max_candidates=50,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
        lime_approximation=True,
        n_samples=10000,
        alpha=1.0,
    )
    action = ce.extract(
        denied[:N], max_change_num=3, cost_type="MPS", tradeoff_parameter=0.75
    )
    if action["feasible"]:
        if not compare:
            for n, r in enumerate(ce.getActionObject(action)):
                print("## {}-th instance:".format(n + 1))
                print(r)
        print("## Summary:")
        print("* Perturbation:")
        for d, a_d in enumerate(action["action"]):
            if abs(a_d) > 1e-8:
                print(
                    "\t* {}: {:+.4f}".format(D.feature_names[d], a_d)
                    if D.feature_types[d] == "C"
                    else "\t* {}: {:+}".format(D.feature_names[d], a_d)
                )
        print("* Score:")
        print("\t* Cost:", action["cost"].mean())
        print("\t* Loss:", (1 - action["active"]).mean())
        print("\t* gamma:", ce.gamma_)
        print("\t* Objective:", action["objective"] / N)
        print("\t* Time[s]:", action["time"])


def __check_lime(dataset="i", model="F", compare=False):
    from utils import DatasetHelper

    np.random.seed(0)

    if model == "L":
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", class_weight="balanced"
        )
    elif model == "F":
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=16, class_weight="balanced"
        )
    elif model == "M":
        from sklearn.neural_network import MLPClassifier

        mdl = MLPClassifier(
            hidden_layer_sizes=(200,), max_iter=500, activation="relu", alpha=0.0001
        )

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    denied = X_tr[mdl.predict(X_tr) == 1]
    denied = denied[denied[:, 20] < 7]

    if compare:
        print("# Without LIME Approximation")
        ce = ActionExtractor(
            mdl,
            X_tr,
            Y=y_tr,
            max_candidates=50,
            feature_names=D.feature_names,
            feature_types=D.feature_types,
            feature_categories=D.feature_categories,
            feature_constraints=D.feature_constraints,
            target_name=D.target_name,
            target_labels=D.target_labels,
            lime_approximation=False,
        )
        action = ce.extract(
            denied, max_change_num=3, cost_type="MPS", tradeoff_parameter=0.75
        )
        if action["feasible"]:
            print("## Summary:")
            print("* Perturbation:")
            for d, a_d in enumerate(action["action"]):
                if abs(a_d) > 1e-8:
                    print(
                        "\t* {}: {:+.4f}".format(D.feature_names[d], a_d)
                        if D.feature_types[d] == "C"
                        else "\t* {}: {:+}".format(D.feature_names[d], a_d)
                    )
            print("* Score:")
            print("\t* Cost:", action["cost"].mean())
            print("\t* Loss:", (1 - action["active"]).mean())
            print("\t* gamma:", ce.gamma_)
            print("\t* Objective:", action["objective"] / denied.shape[0])
            print("\t* Time[s]:", action["time"])
        print()

    print("# With LIME Approximation")
    ce = ActionExtractor(
        mdl,
        X_tr,
        Y=y_tr,
        max_candidates=50,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
        lime_approximation=True,
        n_samples=10000,
        alpha=1.0,
    )
    action = ce.extract(
        denied, max_change_num=3, cost_type="MPS", tradeoff_parameter=1.0
    )
    if action["feasible"]:
        if not compare:
            for n, r in enumerate(ce.getActionObject(action)):
                print("## {}-th instance:".format(n + 1))
                print(r)
        print("## Summary:")
        print("* Perturbation:")
        for d, a_d in enumerate(action["action"]):
            if abs(a_d) > 1e-8:
                print(
                    "\t* {}: {:+.4f}".format(D.feature_names[d], a_d)
                    if D.feature_types[d] == "C"
                    else "\t* {}: {:+}".format(D.feature_names[d], a_d)
                )
        print("* Score:")
        print("\t* Cost:", action["cost"].mean())
        print("\t* Loss:", (1 - action["active"]).mean())
        print("\t* gamma:", ce.gamma_)
        print("\t* Objective:", action["objective"] / denied.shape[0])
        print("\t* Time[s]:", action["time"])


if __name__ == "__main__":
    # _check_ce(N=10, dataset='i', model='L')
    # _check_lime(N=1, dataset='h', model='M', compare=True)
    __check_lime(dataset="i", model="F", compare=False)
