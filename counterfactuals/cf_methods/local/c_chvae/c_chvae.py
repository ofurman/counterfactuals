import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA

from .mlmodel import MLModel
from .utils import (
    check_counterfactuals,
    reconstruct_encoding_constraints,
)
from .vae import VariationalAutoencoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class CCHVAE:
    """Implementation of CCHVAE.

    This class implements the Counterfactuals via Conditional Variational Autoencoders
    (CCHVAE) method for generating model-agnostic counterfactual explanations for
    tabular data, following Pawelczyk et al. (2020).

    Args:
      mlmodel: Black-box model wrapper used for prediction and data access.
      hyperparams: Dictionary of hyperparameters. See Notes for details.

    Notes:
      Hyperparameters (`hyperparams`) control initialization and search behavior:

      - `"data_name"` (str): Name of the dataset.
      - `"n_search_samples"` (int, default: 300): Number of candidate counterfactuals sampled per iteration.
      - `"p_norm"` (int in {1, 2}): L_p norm used for distance calculation.
      - `"step"` (float, default: 0.1): Step size for expanding the search radius.
      - `"max_iter"` (int, default: 2000): Maximum iterations per factual instance.
      - `"clamp"` (bool, default: True): If True, feature values are clamped to [0, 1].
      - `"binary_cat_features"` (bool, default: True): If True, categorical encoding uses drop-if-binary.
      - `"vae_params"` (Dict): Parameters for the VAE:
        - `"layers"` (List[int]): Number of neurons per layer.
        - `"train"` (bool, default: True): Whether to train a new VAE.
        - `"kl_weight"` (float, default: 0.3): KL divergence weight for the VAE loss.
        - `"lambda_reg"` (float, default: 1e-6): Regularization weight for VAE.
        - `"epochs"` (int, default: 5): Training epochs for the VAE.
        - `"lr"` (float, default: 1e-3): Learning rate for the VAE optimizer.
        - `"batch_size"` (int, default: 32): Batch size for VAE training.

    References:
      Pawelczyk, M., Broelemann, K., & Kasneci, G. (2020).
      Learning Model-Agnostic Counterfactual Explanations for Tabular Data.
      In *Proceedings of The Web Conference 2020*.
    """

    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None) -> None:
        """Initializes the CCHVAE method.

        Args:
          mlmodel: Model wrapper providing prediction and dataset utilities.
          hyperparams: Hyperparameter dictionary

        Raises:
          ValueError: If the provided model backend is unsupported.
          FileNotFoundError: If VAE loading is requested but the model file is missing.
        """
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        self._mlmodel = mlmodel
        self._params = hyperparams

        self._n_search_samples = self._params["n_search_samples"]
        self._p_norm = self._params["p_norm"]
        self._step = self._params["step"]
        self._max_iter = self._params["max_iter"]
        self._clamp = self._params["clamp"]

        vae_params = self._params["vae_params"]
        self._generative_model = self._load_vae(
            self._mlmodel.data.df, vae_params, self._mlmodel, self._params["data_name"]
        )

    def _load_vae(
        self, data: pd.DataFrame, vae_params: Dict, mlmodel: MLModel, data_name: str
    ) -> VariationalAutoencoder:
        """Creates or loads the Variational Autoencoder used by CCHVAE.

        If `vae_params["train"]` is True, a new VAE is trained on the provided data.
        Otherwise, an existing VAE is loaded from disk.

        Args:
          data: Full dataset as a DataFrame.
          vae_params: VAE configuration dictionary (see class Notes).
          mlmodel: Model wrapper, used to obtain the mutable feature mask and input order.
          data_name: Name of the dataset for saving/loading the VAE.

        Returns:
          The initialized (and optionally trained/loaded) `VariationalAutoencoder`.

        Raises:
          FileNotFoundError: If loading is requested but the VAE file is not found.
        """
        generative_model = VariationalAutoencoder(
            data_name, vae_params["layers"], mlmodel.get_mutable_mask()
        )

        if vae_params["train"]:
            generative_model.fit(
                xtrain=data[mlmodel.feature_input_order],
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                generative_model.load(vae_params["layers"][0])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

        return generative_model

    def _hyper_sphere_coordindates(
        self, instance, high: int, low: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples points on a p-norm hypersphere shell around an instance.

        The method draws random directions, scales them to lie within the shell
        defined by radii `[low, high)`, and returns the perturbed candidates and
        their sampled distances.

        Args:
          instance: Input point array of shape `(1, d)` or broadcastable to `(n, d)`.
          high: Upper bound (exclusive) of the radius; must be `>= 0` and `high > low`.
          low: Lower bound (inclusive) of the radius; must be `>= 0` and `low < high`.

        Returns:
          Tuple[np.ndarray, np.ndarray]:
            - Candidate counterfactuals as an array of shape `(n_search_samples, d)`.
            - Corresponding distances (radii) as an array of shape `(n_search_samples,)`.

        Raises:
          ValueError: If the configured `p_norm` is not 1 or 2.
        """
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = (
            np.random.rand(self._n_search_samples) * (high - low) + low
        )  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
        return candidate_counterfactuals, dist

    def _counterfactual_search(
        self, step: int, factual: torch.Tensor, cat_features_indices: List
    ) -> pd.DataFrame:
        """Searches for a counterfactual by expanding a hypersphere in latent space.

        Starting from the encoded factual, this method repeatedly samples candidates
        on a growing hypersphere in latent space, decodes them, enforces encoding
        constraints, and checks for label change while minimizing distance under the
        configured p-norm.

        Args:
          step: Increment used to expand the search radius after unsuccessful attempts.
          factual: Single factual instance as a tensor of shape `(1, d)`.
          cat_features_indices: Column indices for encoded categorical features.

        Returns:
          A single counterfactual instance as a 1D NumPy array of shape `(d,)`.

        Raises:
          ValueError: If `p_norm` is not in {1, 2}.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # init step size for growing the sphere
        low = 0
        high = step
        # counter
        count = 0
        counter_step = 1

        torch_fact = torch.from_numpy(factual).to(device)

        # get predicted label of instance
        instance_label = np.argmax(
            self._mlmodel.predict_proba(torch_fact.float()),
            axis=1,
        )

        # vectorize z
        z = self._generative_model.encode(
            torch_fact[:, self._generative_model.mutable_mask].float()
        )[0]
        # add the immutable features to the latents
        z = torch.cat([z, torch_fact[:, ~self._generative_model.mutable_mask]], dim=-1)
        z = z.cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)

        # make copy such that we later easily combine the immutables and the reconstructed mutables
        fact_rep = torch_fact.reshape(1, -1).repeat_interleave(
            self._n_search_samples, dim=0
        )

        candidate_dist: List = []
        x_ce: Union[np.ndarray, torch.Tensor] = np.array([])
        while count <= self._max_iter or len(candidate_dist) <= 0:
            count = count + counter_step
            if count > self._max_iter:
                logger.debug("No counterfactual example found")
                return x_ce[0]

            # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
            latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low)
            torch_latent_neighbourhood = (
                torch.from_numpy(latent_neighbourhood).to(device).float()
            )
            x_ce = self._generative_model.decode(torch_latent_neighbourhood)

            # add the immutable features to the reconstruction
            temp = fact_rep.clone()
            temp[:, self._generative_model.mutable_mask] = x_ce.to(temp.dtype)
            x_ce = temp

            x_ce = reconstruct_encoding_constraints(
                x_ce, cat_features_indices, self._params["binary_cat_features"]
            )
            x_ce = x_ce.detach().cpu().numpy()
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self._p_norm == 1:
                distances = np.abs((x_ce - torch_fact.cpu().detach().numpy())).sum(
                    axis=1
                )
            elif self._p_norm == 2:
                distances = LA.norm(x_ce - torch_fact.cpu().detach().numpy(), axis=1)
            else:
                raise ValueError("Possible values for p_norm are 1 or 2")

            # counterfactual labels
            y_candidate = np.argmax(
                self._mlmodel.predict_proba(torch.from_numpy(x_ce).float()), axis=1
            )
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce[indices]
            candidate_dist = distances[indices]
            # no candidate found & push search range outside
            if len(candidate_dist) == 0:
                low = high
                high = low + step
            elif len(candidate_dist) > 0:
                # certain candidates generated
                min_index = np.argmin(candidate_dist)
                logger.debug("Counterfactual example found")
                return candidate_counterfactuals[min_index]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generates counterfactuals for the given factual instances with validation.

        This method applies the internal search for each factual row, checks the
        validity of found counterfactuals, and returns them in the original feature
        order of the model.

        Args:
          factuals: DataFrame of factual instances.

        Returns:
          DataFrame containing validated counterfactual instances aligned to `factuals`.
        """
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.categorical
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        df_cfs = factuals.apply(
            lambda x: self._counterfactual_search(
                self._step, x.reshape((1, -1)), cat_features_indices
            ),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs

    def get_counterfactuals_without_check(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generates counterfactuals without running the post-hoc validity checks.

        This is similar to `get_counterfactuals` but skips `check_counterfactuals`,
        returning the raw counterfactual outputs projected back to the model's
        original feature order.

        Args:
          factuals: DataFrame of factual instances.

        Returns:
          DataFrame containing counterfactual instances aligned to `factuals`.
        """
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.categorical
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        df_cfs = factuals.apply(
            lambda x: self._counterfactual_search(
                self._step, x.reshape((1, -1)), cat_features_indices
            ),
            raw=True,
            axis=1,
        )

        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
