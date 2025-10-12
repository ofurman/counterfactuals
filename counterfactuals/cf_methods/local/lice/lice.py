from time import perf_counter

import numpy as np
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io import load_onnx_neural_network
from omlt.neuralnet import FullSpaceNNFormulation
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverStatus, TerminationCondition

from counterfactuals.cf_methods.local.lice.data.DataHandler import DataHandler
from counterfactuals.cf_methods.local.lice.data.Types import DataLike
from counterfactuals.cf_methods.local.lice.SPN import SPN

from .data_enc import decode_input_change, encode_input_change
from .spn_enc import encode_spn


class LiCE:
    MIO_EPS = 1e-6

    def __init__(self, spn: SPN, nn_path: str, data_handler: DataHandler) -> None:
        self.__spn = spn
        self.__nn_path = nn_path
        self.__dhandler = data_handler

    # TODO remove the defaults maybe?
    def __build_model(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float,
        optimize_ll: bool,
        prediction_threshold: float = 1e-4,
        ll_opt_coef: float = 0.1,
        leaf_encoding: float = "histogram",
        spn_variant: str = "lower",
    ) -> pyo.Model:
        # Lazy import to avoid dependency issues when LiCE is not used
        import onnx

        model = pyo.ConcreteModel()

        model.input_encoding = pyo.Block()
        inputs, distance = encode_input_change(
            self.__dhandler, model.input_encoding, factual
        )

        model.predictor = OmltBlock()
        onnx_model = onnx.load(self.__nn_path)
        input_bounds = []
        input_vec = []
        for input_var in inputs:
            for var in input_var.values():
                input_vec.append(var)
                input_bounds.append(var.bounds)

        net = load_onnx_neural_network(onnx_model, input_bounds=input_bounds)
        formulation = FullSpaceNNFormulation(net)
        model.predictor.build_formulation(formulation)

        # connect the vars
        model.inputset = pyo.Set(initialize=range(len(input_vec)))

        def connect_input(mdl, i):
            return input_vec[i] == mdl.predictor.inputs[i]

        model.connect_nn_input = pyo.Constraint(model.inputset, rule=connect_input)

        sign = -1 if desired_class == 0 else 1
        model.classification = pyo.Constraint(
            expr=sign * model.predictor.outputs[0] >= prediction_threshold
        )

        # TODO put this to dataenc or to spn, using the fact that spn object knows about features (afaik)
        # spn_inputs = []
        # model.contig_names = pyo.Set(
        #     initialize=[
        #         f.name for f in self.__dhandler.features if isinstance(f, Contiguous)
        #     ]
        # )
        # contig_bounds = {
        #     f.name: f.bounds
        #     for f in self.__dhandler.features
        #     if isinstance(f, Contiguous)
        # }
        # model.spn_input = pyo.Var(
        #     model.contig_names, bounds=contig_bounds, domain=pyo.Reals
        # )

        # def set_scale(m, name: str):
        #     i = self.__dhandler.feature_names.index(name)
        #     f = self.__dhandler.features[i]
        #     return m.spn_input[name] == inputs[i] * f._scale + f._shift

        # model.spn_input_set = pyo.Constraint(model.contig_names, rule=set_scale)
        # for input_var, f in zip(inputs, self.__dhandler.features):
        #     if isinstance(f, Contiguous):
        #         spn_inputs.append(model.spn_input[f.name])
        #     else:
        #         spn_inputs.append(input_var)
        spn_inputs = inputs

        if optimize_ll:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs + [int(desired_class)],
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            model.obj = pyo.Objective(
                expr=distance - ll_opt_coef * spn_outputs[self.__spn.out_node_id],
                sense=pyo.minimize,
            )
            return model

        elif ll_threshold > -np.inf:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs + [int(desired_class)],
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            model.ll_constr = pyo.Constraint(
                expr=spn_outputs[self.__spn.out_node_id] >= ll_threshold
            )

        # set up objective
        model.obj = pyo.Objective(expr=distance, sense=pyo.minimize)
        # model.objconstr = pyo.Constraint(expr=distance == 0)
        # model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        return model

    def generate_counterfactual(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float = -np.inf,
        ll_opt_coefficient: float = 0,
        n_counterfactuals: int = 1,
        solver_name: str = "gurobi",
        verbose: bool = False,
        time_limit: int = 600,
        leaf_encoding: str = "histogram",
        spn_variant: str = "lower",
        ce_relative_distance: float = np.inf,
        ce_max_distance: float = np.inf,
    ) -> tuple[bool, list[DataLike]]:
        t_start = perf_counter()
        model = self.__build_model(
            factual,
            desired_class,
            ll_threshold,
            ll_opt_coefficient != 0,
            leaf_encoding=leaf_encoding,
            ll_opt_coef=ll_opt_coefficient,
            spn_variant=spn_variant,
        )
        t_built = perf_counter()
        if solver_name == "gurobi":
            opt = pyo.SolverFactory(solver_name, solver_io="python")
        else:
            opt = pyo.SolverFactory(solver_name)

        if n_counterfactuals > 1:
            if solver_name != "gurobi":
                raise NotImplementedError(
                    "Generating multiple counterfactuals is supported only for Gurobi solver"
                )
            opt.options["PoolSolutions"] = n_counterfactuals  # Store n solutions
            opt.options["PoolSearchMode"] = 2  # Systematic search for n-best solutions
            if ce_relative_distance != np.inf:
                # Accept solutions within ce_relative_distance*100% of the optimal
                opt.options["PoolGap"] = ce_relative_distance
        if ce_max_distance != np.inf:
            print("Limiting max distance by", ce_max_distance)
            model.max_dist = pyo.Constraint(
                expr=model.input_encoding.total_cost <= ce_max_distance
            )

        if "cplex" in solver_name:
            opt.options["timelimit"] = time_limit
        elif "glpk" in solver_name:
            opt.options["tmlim"] = time_limit
        elif "xpress" in solver_name:
            opt.options["soltimelimit"] = time_limit
            # Use the below instead for XPRESS versions before 9.0
            # self.solver.options['maxtime'] = TIME_LIMIT
        elif "highs" in solver_name:
            opt.options["time_limit"] = time_limit
        elif solver_name == "gurobi":
            opt.options["TimeLimit"] = time_limit
            # opt.options["Aggregate"] = 0
            # opt.options["OptimalityTol"] = 1e-3
            opt.options["IntFeasTol"] = self.MIO_EPS / 10
            opt.options["FeasibilityTol"] = self.MIO_EPS / 10
        else:
            print("Time limit not set! Not implemented for your solver")

        t_prepped = perf_counter()
        result = opt.solve(model, load_solutions=False, tee=verbose)
        t_solved = perf_counter()

        self.__t_build = t_built - t_start
        self.__t_solve = t_solved - t_prepped
        self.__model = model
        self.__loglikelihoods = []
        self.__distances = []

        if verbose:
            opt._solver_model.printStats()
            print(result)
        if result.solver.status == SolverStatus.ok:
            if result.solver.termination_condition == TerminationCondition.optimal:
                # print(pyo.value(model.obj))
                # print(model.spn.node_out[self.__spn.out_node_id].value)
                model.solutions.load_from(result)
                CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
                self.__t_tot = perf_counter() - t_start
                self.__optimal = True
                return CEs
        elif result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
            # the objective value is always bounded
        ]:
            print("Infeasible formulation")
            if verbose:
                write_iis(model, "IIS.ilp", solver="gurobi")
            self.__t_tot = perf_counter() - t_start
            self.__optimal = False
            return []
        elif (
            result.solver.status == SolverStatus.aborted
            and result.solver.termination_condition == TerminationCondition.maxTimeLimit
        ):
            print("TIME LIMIT")
            self.__optimal = False
            try:
                model.solutions.load_from(result)
            except ValueError:
                self.__t_tot = perf_counter() - t_start
                return []
            CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
            self.__t_tot = perf_counter() - t_start
            return CEs
        # else:

        self.__t_tot = (perf_counter() - t_start,)
        self.__optimal = False
        # print result if it wasn't printed yet
        if not verbose:
            print(result)
        raise ValueError("Unexpected termination condition")

    def __get_CEs(
        self, n: int, model: pyo.Model, factual: np.ndarray, opt: pyo.SolverFactory
    ):
        if n > 1:
            # this takes a lot of time for high n (~100 000)
            CEs = []
            self.__loglikelihoods = []
            self.__distances = []
            for sol in range(min(n, opt._solver_model.SolCount)):
                opt._solver_model.params.SolutionNumber = sol
                vars = opt._solver_model.getVars()
                for var in vars:
                    value = var.Xn
                    # or correct some numerical errors
                    # value = np.round(var.Xn, 10)
                    var = opt._solver_var_to_pyomo_var_map[var]
                    if var.bounds != (None, None):
                        value = np.clip(value, var.bounds[0], var.bounds[1])
                    if var.domain in [
                        pyo.Integers,
                        pyo.NonNegativeIntegers,
                        pyo.NonPositiveIntegers,
                        pyo.NegativeIntegers,
                        pyo.PositiveIntegers,
                        pyo.Binary,
                    ]:
                        # value = np.round(value)
                        value = np.round(
                            value, -np.log10(self.MIO_EPS / 10).astype(int)
                        )
                    var.value = value
                self.__distances.append(self.__model.input_encoding.total_cost.value)
                if hasattr(self.__model, "spn"):
                    self.__loglikelihoods.append(
                        self.__model.spn.node_out[self.__spn.out_node_id].value
                    )
                    # TODO move to spn enc?
                CEs.append(
                    decode_input_change(
                        self.__dhandler,
                        model.input_encoding,
                        factual,
                        # round_cont_to=int(-np.log10(self.MIO_EPS)),
                        mio_eps=self.MIO_EPS,
                        spn=self.__spn,
                        mio_spn=(
                            self.__model.spn if hasattr(self.__model, "spn") else None
                        ),
                    )
                )
            return CEs
        else:
            self.__distances.append(self.__model.input_encoding.total_cost.value)
            if hasattr(self.__model, "spn"):
                self.__loglikelihoods.append(
                    self.__model.spn.node_out[self.__spn.out_node_id].value
                )
            return [
                decode_input_change(
                    self.__dhandler,
                    model.input_encoding,
                    factual,
                    # round_cont_to=int(-np.log10(self.MIO_EPS)),
                    mio_eps=self.MIO_EPS,
                    spn=self.__spn,
                    mio_spn=self.__model.spn if hasattr(self.__model, "spn") else None,
                )
            ]

    @property
    def stats(self):
        return {
            "time_total": self.__t_tot,  # with CE recovery
            "time_solving": self.__t_solve,
            "time_building": self.__t_build,
            "optimal": self.__optimal,
            "ll_computed": self.__loglikelihoods,
            "dist_computed": self.__distances,
        }

    @property
    def model(self) -> pyo.Model:
        return self.__model
