import inspect
import logging
import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.ceflow.ceflow import CeFlow, CeFlowParams
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.utils import apply_categorical_discretization

logger = logging.getLogger(__name__)


def _resolve_flow_transforms(flow_model):
    if hasattr(flow_model, "transform_to_latent") and hasattr(flow_model, "transform_to_data"):

        def encode(x):
            return flow_model.transform_to_latent(x)

        def decode(z):
            return flow_model.transform_to_data(z)

        return encode, decode

    if hasattr(flow_model, "inverse"):
        return None, None

    flow_core = getattr(flow_model, "model", None)
    if flow_core is None:
        raise ValueError("CeFlow requires a flow model with an inverse or a .model attribute.")

    if hasattr(flow_core, "transform_to_noise") and hasattr(flow_core, "transform_to_data"):

        def encode(x):
            z_value, _ = flow_core.transform_to_noise(x)
            return z_value

        def decode(z):
            return flow_core.transform_to_data(z)

        return encode, decode

    transform = getattr(flow_core, "_transform", None)
    if transform is None:
        raise ValueError("CeFlow could not find flow transforms on the provided model.")

    def encode(x):
        output = transform.forward(x) if hasattr(transform, "forward") else transform(x)
        return output[0] if isinstance(output, tuple) else output

    def decode(z):
        output = transform.inverse(z) if hasattr(transform, "inverse") else transform(z)
        return output[0] if isinstance(output, tuple) else output

    return encode, decode


def _wrap_with_dequantizer(base_encode, base_decode, flow_model, dequantizer):
    def encode(x):
        x_np = x.detach().cpu().numpy()
        x_dq = dequantizer.transform(x_np)
        x_tensor = torch.from_numpy(x_dq).float().to(x.device)
        if base_encode is not None:
            return base_encode(x_tensor)
        return flow_model(x_tensor)

    def decode(z):
        if base_decode is not None:
            x_tensor = base_decode(z)
        else:
            x_tensor = flow_model.inverse(z)
        x_np = x_tensor.detach().cpu().numpy()
        x_inv = dequantizer.inverse_transform(x_np)
        return torch.from_numpy(x_inv).float().to(z.device)

    return encode, decode


def _build_ceflow_params(cfg):
    return CeFlowParams(
        batch_size=cfg.counterfactuals_params.batch_size,
        alpha_min=cfg.counterfactuals_params.alpha_min,
        alpha_max=cfg.counterfactuals_params.alpha_max,
        alpha_steps=cfg.counterfactuals_params.alpha_steps,
        alpha_grid=list(cfg.counterfactuals_params.alpha_grid),
        distance_metric=cfg.counterfactuals_params.distance_metric,
        binary_logits=cfg.counterfactuals_params.binary_logits,
        clamp_min=cfg.counterfactuals_params.clamp_min,
        clamp_max=cfg.counterfactuals_params.clamp_max,
        use_predicted_labels=cfg.counterfactuals_params.use_predicted_labels,
    )


def _create_gen_model_from_cfg(model_cfg, dataset, model_path, dequantizer):
    model_target = model_cfg.model._target_
    is_ceflow_gmm = "CeFlowGMM" in model_target
    init_kwargs = {"features": dataset.X_train.shape[1]}
    if is_ceflow_gmm:
        init_kwargs["categorical_groups"] = dataset.categorical_features_lists
        init_kwargs["n_classes"] = len(np.unique(dataset.y_train))
    else:
        context_features = model_cfg.get("context_features", 1)
        init_kwargs["context_features"] = context_features
    gen_model = instantiate(model_cfg.model, **init_kwargs)
    if model_cfg.train_model:
        train_loader = dataset.train_dataloader(
            batch_size=model_cfg.batch_size,
            shuffle=True,
            noise_lvl=model_cfg.noise_lvl,
        )
        test_loader = dataset.test_dataloader(
            batch_size=model_cfg.batch_size,
            shuffle=False,
        )
        fit_kwargs = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "epochs": model_cfg.epochs,
            "patience": model_cfg.patience,
            "lr": model_cfg.lr,
            "checkpoint_path": model_path,
        }
        if "dequantizer" in inspect.signature(gen_model.fit).parameters:
            fit_kwargs["dequantizer"] = dequantizer
        gen_model.fit(**fit_kwargs)
        gen_model.save(model_path)
    else:
        gen_model.load(model_path)
    gen_model.eval()
    return gen_model


class CeFlowComposite:
    """Wrapper holding both flow and density models for CeFlow."""

    def __init__(self, flow_model, density_model):
        self.flow_model = flow_model
        self.density_model = density_model


class CeFlowPipelineRunner(PipelineRunner):
    """Pipeline runner for CeFlow counterfactual generation."""

    cf_method_name = "CeFlow"

    def __init__(self, cfg, logger, preprocessing_pipeline=None):
        super().__init__(cfg, logger, preprocessing_pipeline)
        self.flow_model = None

    def create_gen_model(self, dataset, path, dequantizer):
        output_folder = os.path.dirname(path)
        flow_model_name = self.cfg.flow_model.model._target_.split(".")[-1]
        disc_model_name = self._get_disc_model_name()
        if self.cfg.experiment.relabel_with_disc_model:
            flow_model_path = os.path.join(
                output_folder,
                f"flow_model_{flow_model_name}_relabeled_by_{disc_model_name}.pt",
            )
        else:
            flow_model_path = os.path.join(output_folder, f"flow_model_{flow_model_name}.pt")

        self.flow_model = _create_gen_model_from_cfg(
            self.cfg.flow_model, dataset, flow_model_path, dequantizer
        )
        density_model = _create_gen_model_from_cfg(self.cfg.gen_model, dataset, path, dequantizer)

        # Return density_model as the gen_model (used for metrics and log_prob)
        # Flow model is stored in self.flow_model for search_counterfactuals
        return density_model

    def compute_log_prob_threshold(self, gen_model, dataset, dequantizer):
        """Use the density_model (gen_model here) for log_prob threshold calculation."""
        self.logger.info("Calculating log_prob_threshold")
        dataset.X_train = dequantizer.transform(dataset.X_train)
        train_dataloader = dataset.train_dataloader(
            batch_size=self.cfg.counterfactuals_params.batch_size, shuffle=False
        )
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(train_dataloader),
            self.cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)
        self.logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
        return log_prob_threshold

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
        disc_model_name = self._get_disc_model_name()
        target_class = self._get_target_class()

        self.logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        self.logger.info("Creating CeFlow counterfactual model")
        if getattr(self.flow_model, "context_features", None):
            raise ValueError(
                "CeFlow flow_model must be unconditional; set flow_model.context_features to null."
            )
        base_encode, base_decode = _resolve_flow_transforms(self.flow_model)
        if hasattr(self.flow_model, "transform_to_latent") and hasattr(
            self.flow_model, "transform_to_data"
        ):
            encode_fn, decode_fn = base_encode, base_decode
        else:
            dequantizer = GroupDequantizer(dataset.categorical_features_lists)
            dequantizer.fit(dataset.X_train)
            encode_fn, decode_fn = _wrap_with_dequantizer(
                base_encode, base_decode, self.flow_model, dequantizer
            )
        params = _build_ceflow_params(self.cfg)
        cf_method = CeFlow(
            flow_model=self.flow_model,
            disc_model=disc_model,
            params=params,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
        )

        self.logger.info("Handling counterfactual generation")
        y_target = np.full_like(y_test_origin, fill_value=target_class)
        with self._timed_search() as timer:
            explanation_result = cf_method.explain(
                X=X_test_origin,
                y_origin=y_test_origin,
                y_target=y_target,
                X_train=dataset.X_train,
                y_train=dataset.y_train,
            )
        cf_search_time = timer["elapsed"]

        Xs = explanation_result.x_origs
        Xs_cfs = explanation_result.x_cfs
        ys_orig = explanation_result.y_origs
        ys_target = explanation_result.y_cf_targets
        preds = disc_model.predict(Xs_cfs)
        model_returned = preds.reshape(-1) == ys_target.reshape(-1)

        if self.cfg.counterfactuals_params.use_categorical:
            Xs_cfs = apply_categorical_discretization(dataset.categorical_features_lists, Xs_cfs)

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="ceflow_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CeFlowPipelineRunner(cfg, logger, CeFlowPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
