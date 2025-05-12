import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from counterfactuals.cf_methods.lice.lice import LiCE
from counterfactuals.cf_methods.lice.nn_model import NNModel

# trunk-ignore-all(bandit/B301)
# trunk-ignore-all(bandit/B403)

time_limit = int(sys.argv[1])
data_names = [sys.argv[2]]
folds = [int(sys.argv[3])]
median = sys.argv[4] == "median"
quartile = sys.argv[4] == "quartile"
optimize = sys.argv[4] == "optimize"
folder = sys.argv[5]
alpha = 0.1 if len(sys.argv) <= 6 else float(sys.argv[6])

print(time_limit, data_names, folds, median, quartile, optimize, folder)
prefix = f"results/{folder}"
spn_variant = "lower"
leaf_encoding = "histogram"

for data_name in data_names:
    for fold in folds:
        path_base = f"{prefix}/{data_name}/{fold}"
        with open(f"{path_base}/models/spn.pickle", "rb") as f:
            spn = pickle.load(f)
        with open(f"{path_base}/models/dhandler.pickle", "rb") as f:
            dhandler = pickle.load(f)

        nn = NNModel(dhandler.encoding_width(True), [20, 10], 1)
        nn.load(f"{path_base}/models/nn.pt")

        X_test = pd.read_csv(f"{path_base}/data/X_subtest.csv", index_col=0)
        y_test = pd.read_csv(f"{path_base}/data/y_subtest.csv", index_col=0)

        X_train = pd.read_csv(f"{path_base}/data/X_train.csv", index_col=0)
        y_train = pd.read_csv(f"{path_base}/data/y_train.csv", index_col=0)
        train_data = np.concatenate([X_train.values, y_train.values], axis=1)
        lls = spn.compute_ll(train_data)
        median_ll = np.median(lls)
        quartile_ll = np.percentile(lls, 0.25)

        lice = LiCE(
            spn,
            nn_path=f"{path_base}/models/nn.onnx",
            data_handler=dhandler,
        )

        results_median = {}
        results_quartile = {}
        results_optimize = {}
        results_sample = {}
        results_nospn = {}
        for i, sample in X_test.iterrows():
            sample_ll = spn.compute_ll(
                np.concatenate([sample.values, y_test.loc[i].values])
            )[0]
            enc_sample = dhandler.encode(X_test.loc[[i]])
            prediction = nn.predict(enc_sample) > 0

            if optimize:
                t = time.perf_counter()
                optimize_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    ll_opt_coefficient=alpha,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                    leaf_encoding=leaf_encoding,
                    spn_variant=spn_variant,
                )
                tdiff = time.perf_counter() - t
                results_optimize[i] = {
                    "CE": optimize_res,
                    "ll_threshold": None,
                    "stats": lice.stats,
                    "time": tdiff,
                }
                print(f"done optimize {lice.stats['optimal']}, {len(optimize_res)}")
            elif quartile:
                t = time.perf_counter()
                quartile_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    quartile_ll,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                    leaf_encoding=leaf_encoding,
                    spn_variant=spn_variant,
                )
                tdiff = time.perf_counter() - t
                results_quartile[i] = {
                    "CE": quartile_res,
                    "ll_threshold": quartile_ll,
                    "stats": lice.stats,
                    "time": tdiff,
                }
                print(f"done quartile {lice.stats['optimal']}, {len(quartile_res)}")
            elif median:
                t = time.perf_counter()
                median_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    median_ll,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                    leaf_encoding=leaf_encoding,
                    spn_variant=spn_variant,
                )
                tdiff = time.perf_counter() - t
                results_median[i] = {
                    "CE": median_res,
                    "ll_threshold": median_ll,
                    "stats": lice.stats,
                    "time": tdiff,
                }
                print(f"done median {lice.stats['optimal']}, {len(median_res)}")

            else:
                t = time.perf_counter()
                sample_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    min(median_ll, sample_ll),
                    n_counterfactuals=10,
                    time_limit=time_limit,
                    leaf_encoding=leaf_encoding,
                    spn_variant=spn_variant,
                )
                tdiff = time.perf_counter() - t
                results_sample[i] = {
                    "CE": sample_res,
                    "ll_threshold": min(median_ll, sample_ll),
                    "stats": lice.stats,
                    "time": tdiff,
                }
                print(f"done sample {lice.stats['optimal']}, {len(sample_res)}")

                t = time.perf_counter()
                nospn_res = lice.generate_counterfactual(
                    sample,
                    not prediction,
                    n_counterfactuals=10,
                    time_limit=time_limit,
                    leaf_encoding=leaf_encoding,
                    spn_variant=spn_variant,
                )
                tdiff = time.perf_counter() - t
                results_nospn[i] = {
                    "CE": nospn_res,
                    "ll_threshold": None,
                    "stats": lice.stats,
                    "time": tdiff,
                }
                print(f"done nospn {lice.stats['optimal']}, {len(nospn_res)}")
            print(f"done iteration {i}")

        os.makedirs(f"{path_base}/CEs", exist_ok=True)

        if optimize:
            with open(f"{path_base}/CEs/LiCE_optimize.pickle", "wb") as f:
                pickle.dump(results_optimize, f)
        elif quartile:
            with open(f"{path_base}/CEs/LiCE_quartile.pickle", "wb") as f:
                pickle.dump(results_quartile, f)
        elif median:
            with open(f"{path_base}/CEs/LiCE_median.pickle", "wb") as f:
                pickle.dump(results_median, f)
        else:
            with open(f"{path_base}/CEs/LiCE_sample.pickle", "wb") as f:
                pickle.dump(results_sample, f)

            with open(f"{path_base}/CEs/MIO_no_spn.pickle", "wb") as f:
                pickle.dump(results_nospn, f)
