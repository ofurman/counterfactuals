import numpy as np
import pyomo.environ as pyo

from counterfactuals.cf_methods.local.lice.data.DataHandler import DataHandler
from counterfactuals.cf_methods.local.lice.data.Features import (
    Binary,
    Categorical,
    Contiguous,
    Feature,
    Mixed,
    Monotonicity,
)
from counterfactuals.cf_methods.local.lice.data.Types import DataLike
from counterfactuals.cf_methods.local.lice.SPN import SPN


def encode_contiguous(
    mio: pyo.Block, init_val: float, feature: Contiguous, mio_eps: float
) -> tuple[pyo.Var, list[pyo.Var]]:
    mio.cont_change = pyo.Set(initialize=["increase", "decrease"])
    mio.var = pyo.Var(bounds=(0, 1), initialize=init_val)

    if feature.modifiable:
        if feature.monotone == Monotonicity.NONE:
            bounds = {"increase": (0, 1 - init_val), "decrease": (0, init_val)}
        elif feature.monotone == Monotonicity.INCREASING:
            bounds = {"increase": (0, 1 - init_val), "decrease": (0, 0)}
        elif feature.monotone == Monotonicity.DECREASING:
            bounds = {"increase": (0, 0), "decrease": (0, init_val)}
        else:
            raise ValueError(f"Invalid monotonicity value of feature {feature}")
    else:
        bounds = {"increase": (0, 0), "decrease": (0, 0)}

    mio.var_change = pyo.Var(mio.cont_change, bounds=bounds, initialize=0)

    mio.change_constr = pyo.Constraint(
        expr=mio.var
        == init_val + mio.var_change["increase"] - mio.var_change["decrease"]
    )
    # Added to avoid duplicate solutions
    mio.exclusivity = pyo.Var(domain=pyo.Binary, initialize=0)
    mio.is_inc = pyo.Constraint(expr=mio.var_change["increase"] <= mio.exclusivity)
    mio.is_dec = pyo.Constraint(
        expr=mio.var_change["decrease"] <= (1 - mio.exclusivity)
    )
    mio.fix_to_zero = pyo.Constraint(
        expr=mio.var_change["decrease"] + mio.var_change["increase"]
        >= mio_eps * mio.exclusivity
    )
    # ---end--- Added to avoid duplicate solutions

    if feature.discrete:
        mio.disc_shadow = pyo.Var(domain=pyo.Integers, bounds=feature.bounds)
        mio.discrete_fix = pyo.Constraint(
            expr=mio.var * feature._scale + feature._shift == mio.disc_shadow
        )

    return mio.var, [mio.var_change[v] for v in mio.cont_change]


def encode_binary(
    mio: pyo.Block, init_val: int, feature: Binary
) -> tuple[pyo.Var, list[pyo.Var]]:
    mio.var = pyo.Var(domain=pyo.Binary, initialize=init_val)
    mio.var_change = pyo.Var(domain=pyo.Binary, initialize=0)
    if feature.modifiable:
        if init_val == 0:
            if feature.monotone == Monotonicity.DECREASING:
                mio.change_constr = pyo.Constraint(expr=mio.var == 0)
            else:
                mio.change_constr = pyo.Constraint(expr=mio.var == mio.var_change)
        if init_val == 1:
            if feature.monotone == Monotonicity.INCREASING:
                mio.change_constr = pyo.Constraint(expr=mio.var == 1)
            else:
                mio.change_constr = pyo.Constraint(expr=mio.var == 1 - mio.var_change)
    else:
        mio.change_constr = pyo.Constraint(expr=mio.var == init_val)
        mio.no_change = pyo.Constraint(expr=mio.var_change == 0)

    return mio.var, [mio.var_change]


def encode_categorical(
    mio: pyo.Block, init_val: int, feature: Categorical, ohe_extra: pyo.Var | int = 0
) -> tuple[pyo.Var, list[pyo.Var]]:
    mio.vals = pyo.Set(initialize=feature.numeric_vals)
    mio.var = pyo.Var(
        mio.vals,
        domain=pyo.Binary,
        initialize={v: int(v == init_val) for v in mio.vals},
    )
    mio.ohe_constr = pyo.Constraint(
        expr=sum(mio.var[v] for v in mio.vals) + ohe_extra == 1
    )

    mio.var_change = pyo.Var(mio.vals, domain=pyo.Binary, initialize=0)
    if feature.modifiable:
        if feature.monotone == Monotonicity.INCREASING:
            mio.inaccessible = pyo.Set(initialize=feature.lower_than(init_val))
        elif feature.monotone == Monotonicity.DECREASING:
            mio.inaccessible = pyo.Set(initialize=feature.greater_than(init_val))
        else:
            mio.inaccessible = pyo.Set(initialize=[])
        mio.modif_constr = pyo.Constraint(
            mio.inaccessible, rule=lambda m, v: (m.var_change[v] == 0)
        )
    else:
        mio.modif_constr = pyo.Constraint(
            mio.vals, rule=lambda m, v: (m.var_change[v] == 0)
        )
    mio.change_constr = pyo.Constraint(
        mio.vals,
        rule=lambda m, v: (
            m.var[v] == 1 - m.var_change[v]
            if v == init_val
            else m.var[v] == m.var_change[v]
        ),
    )
    return mio.var, [mio.var_change[v] for v in mio.vals]


def encode_mixed(
    mio: pyo.Block, init_val: int, feature: Mixed
) -> tuple[tuple[pyo.Var, list[pyo.Var]], tuple[pyo.Var, list[pyo.Var]]]:
    mio.contiguous = pyo.Block()
    if init_val in feature.numeric_vals:
        cont_val = feature.default_val_normalized
    else:
        cont_val = init_val
    cont_res = encode_contiguous(mio.contiguous, cont_val)

    mio.categorical = pyo.Block()
    mio.is_cont = pyo.Var(
        domain=pyo.Binary, initialize=int(init_val not in feature.numeric_vals)
    )
    categ_res = encode_categorical(
        mio.categorical,
        init_val,
        feature,
        mio.is_cont,
    )
    # TODO monotonicity? how to specify
    return cont_res, categ_res


def encode_causal_increase(
    mio: pyo.Block,
    cause: Feature,
    cause_init: int | float,
    cause_mio: pyo.Block,
    effect: Feature,
    effect_init: int | float,
    effect_mio: pyo.Block,
    mio_eps=float,
):
    mio.activated = pyo.Var(domain=pyo.Binary, initialize=0)
    if isinstance(cause, Categorical):
        mio.has_increased = pyo.Constraint(
            expr=mio.activated
            == sum([cause_mio.var[i] for i in cause.greater_than(cause_init)])
        )
    elif isinstance(cause, Contiguous):
        mio.has_increased = pyo.Constraint(
            expr=mio.activated >= cause_mio.var_change["increase"]
        )
        # Added to avoid duplicate solutions
        mio.fix_to_zero = pyo.Constraint(
            expr=mio.activated * mio_eps <= cause_mio.var_change["increase"]
        )
    else:
        ValueError("Other feature types are not supported")

    if isinstance(effect, Categorical):
        mio.must_increase = pyo.Constraint(
            expr=mio.activated
            <= sum([effect_mio.var[i] for i in effect.greater_than(effect_init)])
        )
    elif isinstance(effect, Contiguous):
        mio.must_increase = pyo.Constraint(
            expr=mio.activated * mio_eps <= effect_mio.var_change["increase"]
        )
    else:
        ValueError("Other feature types are not supported")


def encode_input_change(
    data_handler: DataHandler,
    mio_block: pyo.Block,
    init_vals: DataLike,  # original values, unencoded
    # cost vectors (mutliple values for categorical) or a string
    change_cost: list[np.ndarray] | str = "inverse_MAD",
    mio_eps: float = 1e-4,
) -> tuple[list[pyo.Var], pyo.Var]:
    """
    Creates an encoding of input features to the [0, 1] interval
    Encodes categorical features as one-hot, binary as a single 0-1 variable.
    Also allows for mixed encoding - continuous with some categorical values (as per Russel 2019)

    mio_epsilon is the minimal change between values - for numerical stability - used here for sharp inequalities

    returns the input variables and a variable that represents the cost
    """
    if change_cost == "inverse_MAD":
        change_cost = []
        for feature in data_handler.features:
            MADs = feature.MAD
            MADs[MADs < mio_eps] = mio_eps
            change_cost.append(1.0 / MADs)
    else:
        if not isinstance(change_cost, list):
            raise ValueError("Unsupported changed cost specification")

    if init_vals.shape[0] != data_handler.n_features:
        raise ValueError(
            "The number of initial values is different from the number of features"
        )
    if len(change_cost) != data_handler.n_features:
        raise ValueError(
            "The number of cost vectors is different from the number of features"
        )

    for i, feat in enumerate(data_handler.features):
        if change_cost[i].shape[0] != feat.encoding_width(
            one_hot=not isinstance(feat, Binary)
        ):
            raise ValueError(
                f"The length of cost vectors does not fit encoded width of feature {feat}"
            )

    normalized = data_handler.encode(init_vals, normalize=True, one_hot=False)

    input_vars = []
    cost_pairs = []
    mio_block.feature_blocks = pyo.Block(data_handler.feature_names)
    for cost, init_val, feature in zip(change_cost, normalized, data_handler.features):
        if isinstance(feature, Contiguous):
            var, changes = encode_contiguous(
                mio_block.feature_blocks[feature.name], init_val, feature, mio_eps
            )
            cost = [cost[0], cost[0]]
        elif isinstance(feature, Binary):
            var, changes = encode_binary(
                mio_block.feature_blocks[feature.name], init_val, feature
            )
        elif isinstance(feature, Categorical):
            var, changes = encode_categorical(
                mio_block.feature_blocks[feature.name], init_val, feature
            )
        elif isinstance(feature, Mixed):
            (cont_var, cont_changes), categ = encode_mixed(
                mio_block.feature_blocks[feature.name], init_val, feature
            )
            # handle cont results here
            input_vars.append(cont_var)
            cost_pairs += [(cost[0], ch) for ch in cont_changes]
            # and the categ results at the end of the loop
            var, changes = categ
            cost = cost[1:]
        else:
            raise NotImplementedError(
                "Encoding of feature type " + str(feature) + " is not handled."
            )
        input_vars.append(var)
        cost_pairs += list(zip(changes, cost))

    mio_block.total_cost = pyo.Var(initialize=0)
    mio_block.cost_constr = pyo.Constraint(
        expr=mio_block.total_cost == sum(v * cost for cost, v in cost_pairs)
    )

    mio_block.causal_set = pyo.Set(
        initialize=[(i.name, j.name) for i, j in data_handler.causal_inc]
    )
    mio_block.causal = pyo.Block(mio_block.causal_set)
    for cause_f, effect_f in data_handler.causal_inc:
        encode_causal_increase(
            mio_block.causal[(cause_f.name, effect_f.name)],
            cause_f,
            normalized[data_handler.features.index(cause_f)],
            mio_block.feature_blocks[cause_f.name],
            effect_f,
            normalized[data_handler.features.index(effect_f)],
            mio_block.feature_blocks[effect_f.name],
            mio_eps,
        )

    mio_block.gt_set = pyo.Set(
        initialize=[(i.name, j.name) for i, j in data_handler.greater_than]
    )

    def ge_constraint(m, greater, smaller):
        # greater, smaller = pair
        print(smaller, greater)
        smaller_f = data_handler.features[data_handler.feature_names.index(smaller)]
        greater_f = data_handler.features[data_handler.feature_names.index(greater)]
        #  shuffle the coefficients to make their values as close to 0-1 range as possible
        return (
            m.feature_blocks[smaller].var
            <= m.feature_blocks[greater].var * (greater_f._scale / smaller_f.scale)
            + (greater_f._shift - smaller_f._shift) / smaller_f.scale
        )

    mio_block.greater_than = pyo.Constraint(mio_block.gt_set, rule=ge_constraint)

    return input_vars, mio_block.total_cost


def decode_input_change(
    data_handler: DataHandler,
    mio_block: pyo.Block,
    factual: DataLike,
    # round_cont_to: float = np.inf,
    mio_eps: float,
    spn: SPN,
    mio_spn: pyo.Block | None,
) -> np.ndarray:
    res = []
    for feature, val in zip(data_handler.features, factual):
        var_change = mio_block.feature_blocks[feature.name].var_change
        if isinstance(feature, Contiguous):
            value = (
                feature.encode(val, normalize=True)
                + var_change["increase"].value
                - var_change["decrease"].value
            )
            # np.round(value, int(-np.log10(mio_eps)))
            # if mio_spn is not None:
            # for node in spn.nodes:
            #     if (
            #         hasattr(node, "scope")
            #         and node.scope
            #         < data_handler.n_features  # target feature is not relevant
            #         and feature == data_handler.features[node.scope]
            #     ):
            #         histogram = mio_spn.find_component(f"HistLeaf{node.id}")
            #         breaks, _ = node.get_breaks_densities(span_all=True)
            #         # omit the last bin, the upper bound is not strict
            #         for bin in range(len(breaks) - 2):
            #             if histogram.not_in_bin[bin].value == 0:
            #                 value = np.clip(
            #                     value, breaks[bin], breaks[bin + 1] - 1e-10
            #                 )
            #                 break
            #         else:
            #             if histogram.not_in_bin[len(breaks) - 2].value != 0:
            #                 raise ValueError("No 0 bin")
            # for node in reversed(spn.nodes):
            #     relevant_nodes = [spn.out_node_id]
            #     if node.id in relevant_nodes:
            #         if hasattr(node, "predecessors"):
            #             slacks = mio_spn.find_component(
            #                 f"SumSlackIndicators{node.id}"
            #             )
            #             if slacks is None:
            #                 relevant_nodes += [p.id for p in node.predecessors]
            #                 continue
            #             for p in node.predecessors:
            #                 if slacks[p.id].value == 0:
            #                     relevant_nodes.append(p.id)
            #         elif (
            #             hasattr(node, "scope")
            #             and node.scope
            #             < data_handler.n_features  # target feature is not relevant
            #             and feature == data_handler.features[node.scope]
            #         ):
            #             histogram = mio_spn.find_component(f"HistLeaf{node.id}")
            #             breaks, _ = node.get_breaks_densities(span_all=True)
            #             # omit the last bin, the upper bound is not strict
            #             for bin in range(len(breaks) - 2):
            #                 if histogram.not_in_bin[bin].value == 0:
            #                     value = np.clip(
            #                         value, breaks[bin], breaks[bin + 1] - 1e-10
            #                     )
            #                     break
            #             else:
            #                 if histogram.not_in_bin[len(breaks) - 2].value != 0:
            #                     raise ValueError("No 0 bin")

            value = feature.decode(
                value,
                denormalize=True,
                return_series=False,
                discretize=True,
            )
            round_cont_to = int(-np.log10(mio_eps))
            value = np.round(value, round_cont_to)
            res.append(value)
        elif isinstance(feature, Binary):
            res.append(
                feature.decode(
                    np.abs(
                        np.round(
                            np.array([feature.encode(val, one_hot=False)])
                            - var_change.value
                        )
                    ),
                    return_series=False,
                )[0]
            )
        elif isinstance(feature, Categorical):
            res.append(
                feature.decode(
                    np.abs(
                        np.round(
                            [
                                np.array([ch.value for ch in var_change.values()])
                                - feature.encode(val, one_hot=True)
                            ]
                        )
                    ),
                    return_series=False,
                )[0]
            )
        elif isinstance(feature, Mixed):
            cont_ch = mio_block.feature_blocks[feature.name].contiguous.var_change
            categ_ch = mio_block.feature_blocks[feature.name].categorical.var_change
            val_vec = feature.encode(val, normalize=True, one_hot=True)
            cont = val[0] + cont_ch["increase"].value - cont_ch["decrease"].value
            categ = np.abs(
                np.round(np.array([ch.value for ch in categ_ch.values()]) - val_vec)
            )
            res.append(
                feature.decode(np.concatenate([cont, categ]), return_series=False)
            )[0]
        else:
            raise NotImplementedError(
                "Encoding of feature type " + str(feature) + " is not handled."
            )
    return np.array(res, dtype=object)
