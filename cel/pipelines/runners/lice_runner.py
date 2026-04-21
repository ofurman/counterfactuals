"""Pipeline runner for LiCE counterfactual generation."""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cel.cf_methods.local_methods.lice.lice import LiCE
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import PipelineRunner, SearchResult
from cel.pipelines.nodes.disc_model_nodes import create_disc_model
from cel.pipelines.nodes.gen_model_nodes import create_gen_model
from cel.pipelines.nodes.helper_nodes import set_model_paths

logger = logging.getLogger(__name__)


class LiCEPipelineRunner(PipelineRunner):
    """Pipeline runner for LiCE counterfactual generation.

    LiCE uses a raw dataset without MethodDataset preprocessing, does not use
    dequantization, and requires SPN and ONNX export. It overrides run() to
    handle these custom requirements.
    """

    cf_method_name = "LiCE"

    def run(self) -> None:
        """Custom run implementation for LiCE with raw dataset and no preprocessing."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.logger.info("Loading dataset")
        dataset = instantiate(self.cfg.dataset, shuffle=False)

        for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
            disc_model_path, gen_model_path, save_folder = set_model_paths(self.cfg, fold=fold_n)
            self.logger.info("Processing fold %d", fold_n)
            disc_model = create_disc_model(self.cfg, dataset, disc_model_path, save_folder)

            if self.cfg.experiment.relabel_with_disc_model:
                dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
                dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

            gen_model = create_gen_model(self.cfg, dataset, gen_model_path)

            result = self.search_counterfactuals(dataset, gen_model, disc_model, save_folder, None)

            metrics = self.calculate_metrics(gen_model, disc_model, dataset, result, None)

            self.save_results(metrics, result.cf_search_time, save_folder)

    def search_counterfactuals(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        save_folder: str,
        log_prob_threshold: float,
    ) -> SearchResult:
        """Generate counterfactuals for the current fold.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model.
            disc_model: Trained discriminative model.
            save_folder: Directory for saving generated counterfactuals.
            log_prob_threshold: Plausibility threshold from compute_log_prob_threshold.

        Returns:
            SearchResult with counterfactuals and timing information.
        """
        disc_model.eval()
        disc_model_name = self._get_disc_model_name()
        X_train, y_train = dataset.X_train, dataset.y_train
        X_test = dataset.X_test

        self.logger.info("Filtering out target class data for counterfactual generation")
        target_class = 1
        ys_pred = disc_model.predict(X_test)
        Xs = dataset.X_test[ys_pred != target_class]
        ys_orig = ys_pred[ys_pred != target_class]

        self.logger.info("Creating counterfactual model")
        # Convert data to pandas DataFrame for LiCE
        X_train_df = pd.DataFrame(X_train, columns=dataset.features[:-1])
        y_train_df = pd.DataFrame(y_train, columns=[dataset.features[-1]])

        # Create data handler and SPN
        from cel.cf_methods.local_methods.lice.data.DataHandler import DataHandler
        from cel.cf_methods.local_methods.lice.SPN import SPN

        dhandler = DataHandler(X_train_df, y_train_df)
        spn = SPN(
            np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
            dhandler,
            normalize_data=True,
        )

        # Create and save ONNX model
        os.makedirs(f"{save_folder}/models", exist_ok=True)
        dummy_input = torch.randn(1, X_train.shape[1])
        torch.onnx.export(
            disc_model,
            dummy_input,
            f"{save_folder}/models/nn.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        lice = LiCE(
            spn,
            nn_path=f"{save_folder}/models/nn.onnx",
            data_handler=dhandler,
        )

        self.logger.info("Calculating log_prob_threshold")
        train_data = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        lls = spn.compute_ll(train_data)
        log_prob_threshold = np.median(lls)
        self.logger.info("log_prob_threshold: %.4f", log_prob_threshold)

        self.logger.info("Handling counterfactual generation")
        with self._timed_search() as timer:
            Xs_cfs = []
            model_returned = []
            ys_target = []
            for i, sample in enumerate(Xs):
                try:
                    enc_sample = dhandler.encode(
                        pd.DataFrame([sample], columns=dataset.features[:-1])
                    )
                    prediction = disc_model.predict(enc_sample) > 0

                    # Generate counterfactual
                    time_limit = 600  # Default time limit in seconds
                    if hasattr(self.cfg, "counterfactuals_params") and hasattr(
                        self.cfg.counterfactuals_params, "time_limit"
                    ):
                        time_limit = self.cfg.counterfactuals_params.time_limit

                    cf = lice.generate_counterfactual(
                        sample,
                        not prediction,
                        ll_threshold=log_prob_threshold,
                        n_counterfactuals=1,
                        time_limit=time_limit,
                        leaf_encoding="histogram",
                        spn_variant="lower",
                        solver_name="cbc",
                    )

                    self.logger.info("Counterfactual: %s", cf)
                    if len(cf) > 0:
                        Xs_cfs.append(cf[0])
                        model_returned.append(True)
                        ys_target.append(1 - prediction)
                    else:
                        Xs_cfs.append(sample)
                        model_returned.append(False)
                        ys_target.append(1 - prediction)
                except Exception as e:
                    self.logger.error(
                        "Error generating counterfactual for sample %d: %s", i, str(e)
                    )
                    Xs_cfs.append(sample)
                    model_returned.append(False)
                    ys_target.append(1 - int(ys_orig[i]))

            Xs_cfs = np.array(Xs_cfs)
            model_returned = np.array(model_returned)
            ys_target = np.array(ys_target)
        cf_search_time = timer["elapsed"]

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs, columns=dataset.features[:-1]).to_csv(
            counterfactuals_path, index=False
        )
        self.logger.info("Counterfactuals saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="lice_config", version_base="1.2")
def main(cfg: DictConfig):
    """Run LiCE pipeline with custom dataset handling."""
    torch.manual_seed(0)
    runner = LiCEPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
