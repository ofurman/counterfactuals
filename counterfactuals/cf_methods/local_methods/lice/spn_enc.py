import numpy as np
import pyomo.environ as pyo

from counterfactuals.cf_methods.local.lice.SPN import SPN, NodeType

# from scipy.special import logsumexp


# issues with binding variables in lambda functions for constraints
# trunk-ignore-all(ruff/B023)


# def contains_positive_logdensities(spn: SPN) -> bool:
#     """Checks whether there is a possibility that the SPN will have input log density on a sum node > 0

#     Args:
#         spn (SPN): The SPN in question

#     Returns:
#         bool: True if maximal log density of inputs to a sum node is greater than 0, False otherwise
#     """
#     node_maxes = {}
#     positive_dens = False
#     for node in spn.nodes:
#         if node.type in [
#             NodeType.LEAF,
#             NodeType.LEAF_BINARY,
#             NodeType.LEAF_CATEGORICAL,
#         ]:
#             node_maxes[node.id] = max(np.log(node.densities))
#         elif node.type == NodeType.PRODUCT:
#             node_maxes[node.id] = sum(node_maxes[n.id] for n in node.predecessors)
#         elif node.type == NodeType.SUM:
#             for pred, w in zip(node.predecessors, node.weights):
#                 if node_maxes[pred.id] + np.log(w) > 0:
#                     positive_dens = True
#             node_maxes[node.id] = logsumexp(
#                 np.array(node_maxes[n.id] for n in node.predecessors), node.weights
#             )
#         else:
#             raise ValueError("Unknown node type")
#     return positive_dens


def encode_histogram_as_pwl(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    encoding_type: str = "LOG",
) -> pyo.Piecewise:
    breakpoints = [breaks[0]]
    for b in breaks[1:-1]:
        breakpoints += [b, b]
    breakpoints.append(breaks[-1])

    doubled_vals = []
    for d in vals:
        doubled_vals += [d, d]

    return pyo.Piecewise(
        out_var,
        in_var,
        pw_pts=breakpoints,
        pw_constr_type="EQ",
        pw_repn=encoding_type,
        f_rule=list(doubled_vals),
    )


def encode_histogram(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    mio_block: pyo.Block,
    mio_epsilon: float,
):
    n_bins = len(vals)
    M = max(1, breaks[-1] - breaks[0])

    mio_block.bins = pyo.Set(initialize=list(range(n_bins)))
    mio_block.not_in_bin = pyo.Var(mio_block.bins, domain=pyo.Binary)
    mio_block.one_bin = pyo.Constraint(
        expr=sum(mio_block.not_in_bin[i] for i in mio_block.bins) == n_bins - 1
    )

    mio_block.lower = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M >= breaks[bin_i] - in_var,
    )
    mio_block.upper = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M >= in_var - breaks[bin_i + 1] + mio_epsilon,
    )

    mio_block.output = pyo.Constraint(
        expr=sum((1 - mio_block.not_in_bin[i]) * vals[i] for i in range(n_bins)) == out_var
    )


def encode_spn(
    spn: SPN,
    mio_spn: pyo.Block,
    input_vars: list[list[pyo.Var] | pyo.Var],
    leaf_encoding: str = "histogram",
    mio_epsilon: float = 1e-6,
    sum_approx: str = "lower",
) -> pyo.Var:
    """Encodes the spn into MIP formulation computing log-likelihood over the input variables

    Args:
        spn (SPN): The SPN to model
        mio_spn (pyo.Block): Pyomo block used to model the spn
        input_vars (list[list[pyo.Var]  |  pyo.Var]): Variables representing the SPN inputed. List of variables (possibly one-hot encoded in the same ordering as SPN bins)
        leaf_encoding (str, optional): either "histogram" or one of the values for piece-wise linear function approximation within Pyomo library (see https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions). Defaults to "histogram".
        mio_epsilon (float, optional): the minimal change between values (for numerical stability), used for sharp inequalities. Defaults to 1e-6.
        TODO add sum_approx

    # Raises:
    #     AssertionError: _description_
    #     ValueError: _description_

    Returns:
        pyo.Var: Indexed pyomo variable containing node outputs, indexed by node ids
    """
    node_ids = [node.id for node in spn.nodes]

    # node_type_ids = {t: [] for t in NodeType}
    # for node in spn.nodes:
    #     node_type_ids[node.type].append(node.id)
    #     node_ids.append(node.id)

    # mio_spn.node_type_sets = {
    #     t: pyo.Set(initialize=ids) for t, ids in node_type_ids.items()
    # }
    mio_spn.node_set = pyo.Set(initialize=node_ids)

    # values are log likelihoods - almost always negative - except in narrow peaks that go above 1
    # mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.NonPositiveReals)
    mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.Reals)
    # print(mio_spn.node_set, node_ids)

    # TODO nodes as blocks
    for node in spn.nodes:
        if node.type == NodeType.LEAF:
            # in_var = mio_spn.input[node.scope]
            in_var = input_vars[node.scope]

            # lb, ub = in_var.bounds

            # if lb is None or ub is None:
            #     raise AssertionError("SPN input variables must have fixed bounds.")

            # density_vals = node.densities
            # breakpoints = node.breaks
            # # if histogram is narrower than the input bounds
            # if lb < breakpoints[0]:
            #     breakpoints = [lb] + breakpoints
            #     density_vals = [spn.min_density] + density_vals
            # if ub > breakpoints[-1]:
            #     breakpoints = breakpoints + [ub]
            #     density_vals = density_vals + [spn.min_density]

            breakpoints, densities = node.get_breaks_densities(span_all=True)
            log_densities = np.log(densities)

            if leaf_encoding == "histogram":
                hist_block = pyo.Block()
                mio_spn.add_component(f"HistLeaf{node.id}", hist_block)
                encode_histogram(
                    breakpoints,
                    log_densities,
                    in_var,
                    mio_spn.node_out[node.id],
                    hist_block,
                    mio_epsilon,  # * spn.input_scale(node.scope),
                )
            else:
                pw_constr = encode_histogram_as_pwl(
                    breakpoints,
                    log_densities,
                    in_var,
                    mio_spn.node_out[node.id],
                    leaf_encoding,
                )
                mio_spn.add_component(f"PWLeaf{node.id}", pw_constr)

        elif node.type == NodeType.LEAF_CATEGORICAL:
            dens_ll = np.log(node.densities)
            in_vars = input_vars[node.scope]

            if isinstance(in_vars, pyo.Var):
                in_vars = [in_vars[k] for k in sorted(in_vars.keys())]

            if len(in_vars) <= 1:  # TODO make this more direct, not fixed to 1
                raise ValueError(
                    "The categorical values should be passed as a list of binary variables, representing a one-hot encoding."
                )
            # Do checks that the vars are binary?
            # check if the histogram always contains all values?
            # TODO use expr parameter of Constraint maker, instead of the rule=lambdas?

            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id] == sum(var * dens for var, dens in zip(in_vars, dens_ll))
                )
            )
            mio_spn.add_component(f"CategLeaf{node.id}", constr)
        elif node.type == NodeType.LEAF_BINARY:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == (1 - input_vars[node.scope]) * np.log(node.densities[0])
                    + input_vars[node.scope] * np.log(node.densities[1])
                )
            )
            mio_spn.add_component(f"BinLeaf{node.id}", constr)
        elif node.type == NodeType.PRODUCT:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id] == sum(b.node_out[ch.id] for ch in node.predecessors)
                )
            )
            mio_spn.add_component(f"ProdConstr{node.id}", constr)
        elif node.type == NodeType.SUM:
            # Sum node - approximated in log domain by max
            preds_set = [ch.id for ch in node.predecessors]
            n_preds = len(node.predecessors)
            weights = {ch.id: w for ch, w in zip(node.predecessors, node.weights)}

            # TODO testing this, if it works well, fit it in correctly
            M_sum = 100  # hope this is enough
            slack_inds = pyo.Var(preds_set, domain=pyo.Binary)
            mio_spn.add_component(f"SumSlackIndicators{node.id}", slack_inds)
            if sum_approx == "lower":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id] + np.log(weights[pre_id]) + M_sum * slack_inds[pre_id]
                    ),
                )
            elif sum_approx == "upper":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id]
                        + (  # approximate by the bound on logsumexp
                            np.log(weights[pre_id] * n_preds)
                            if weights[pre_id] * n_preds < 1
                            else 0  # or by using the fact it is a mixture
                        )
                        + M_sum * slack_inds[pre_id]
                    ),
                )
            else:
                raise ValueError('sum_approx must be one of ["upper", "lower"]')
            mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)
            one_tight = pyo.Constraint(expr=sum(slack_inds[i] for i in preds_set) == n_preds - 1)
            mio_spn.add_component(f"SumTightConstr{node.id}", one_tight)

            # implemented using SOS1 constraints, see here: https://www.gurobi.com/documentation/current/refman/general_constraints.html
            # slacks = pyo.Var(preds_set, domain=pyo.NonNegativeReals)
            # mio_spn.add_component(f"SumSlackVars{node.id}", slacks)
            # if sum_approx == "lower":
            #     slacking = pyo.Constraint(
            #         preds_set,
            #         rule=lambda b, pre_id: (
            #             b.node_out[node.id]
            #             == b.node_out[pre_id] + np.log(weights[pre_id]) + slacks[pre_id]
            #         ),
            #     )
            # elif sum_approx == "upper":
            #     slacking = pyo.Constraint(
            #         preds_set,
            #         rule=lambda b, pre_id: (
            #             b.node_out[node.id]
            #             == b.node_out[pre_id]
            #             + (  # approximate by the bound on logsumexp
            #                 np.log(weights[pre_id] * n_preds)
            #                 if weights[pre_id] * n_preds < 1
            #                 else 0  # or by using the fact it is a mixture
            #             )
            #             + slacks[pre_id]
            #         ),
            #     )
            # else:
            #     raise ValueError('sum_approx must be one of ["upper", "lower"]')
            # mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)

            # indicators = pyo.Var(preds_set, domain=pyo.Binary)
            # mio_spn.add_component(f"SumIndicators{node.id}", indicators)
            # indicating = pyo.Constraint(
            #     rule=lambda b: (
            #         sum(b.component(f"SumIndicators{node.id}")[i] for i in preds_set)
            #         == 1
            #     )
            # )
            # mio_spn.add_component(f"SumIndicatorConstr{node.id}", indicating)

            # sos = pyo.SOSConstraint(
            #     preds_set,
            #     rule=lambda b, pred: [
            #         b.component(f"SumIndicators{node.id}")[pred],
            #         b.component(f"SumSlackVars{node.id}")[pred],
            #     ],
            #     sos=1,
            # )
            # mio_spn.add_component(f"SumSosConstr{node.id}", sos)

    return mio_spn.node_out
