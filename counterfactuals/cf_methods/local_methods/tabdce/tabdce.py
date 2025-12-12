import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)

# Zakładam importy Twoich klas
from .diffusion import MixedTabularDiffusion
from .data import TabularCounterfactualDataset

class TabDCE(BaseCounterfactualMethod):
    def __init__(
        self,
        diffusion_model: MixedTabularDiffusion,
        spec,       
        qt,        
        ohe,       
        device="cpu"
    ):
        self.diffusion_model = diffusion_model
        self.spec = spec
        self.qt = qt
        self.ohe = ohe
        self.device = device
        
        self.diffusion_model.eval()
        self.diffusion_model.to(self.device)

    def _transform_to_model_space(self, X_numpy: np.ndarray) -> torch.Tensor:
        N = X_numpy.shape[0]
        
        if len(self.spec.num_idx) > 0:
            X_num = X_numpy[:, self.spec.num_idx].astype(np.float32)
            if self.qt is not None:
                X_num_tr = self.qt.transform(X_num)
            else:
                X_num_tr = X_num
        else:
            X_num_tr = np.zeros((N, 0), dtype=np.float32)

        if len(self.spec.cat_idx) > 0:
            X_cat = X_numpy[:, self.spec.cat_idx]
            if self.ohe is not None:
                X_cat_oh = self.ohe.transform(X_cat)
                X_cat_log = np.log(np.clip(X_cat_oh, 1e-30, 1.0)).astype(np.float32)
            else:
                X_cat_log = np.zeros((N, 0), dtype=np.float32)
        else:
            X_cat_log = np.zeros((N, 0), dtype=np.float32)

        X_model = np.concatenate([X_num_tr, X_cat_log], axis=1)
        return torch.from_numpy(X_model).float().to(self.device)

    def _inverse_transform_from_model_space(self, X_model_tensor: torch.Tensor) -> np.ndarray:
        X_model = X_model_tensor.detach().cpu().numpy()
        N = X_model.shape[0]
        
        num_dim = self.diffusion_model.num_numerical
        
        if num_dim > 0:
            X_num_tr = X_model[:, :num_dim]
            X_num_tr = np.clip(X_num_tr, -5.2, 5.2)
            if self.qt is not None:
                X_num_orig = self.qt.inverse_transform(X_num_tr)
            else:
                X_num_orig = X_num_tr
        else:
            X_num_orig = np.zeros((N, 0))

        if self.ohe is not None:
            X_cat_part = X_model[:, num_dim:]

            reconstructed_cols = []
            start = 0
            cat_lens = [len(c) for c in self.ohe.categories_]
            
            for i, k in enumerate(cat_lens):
                part = X_cat_part[:, start:start+k]
                indices = np.argmax(part, axis=1) 
                
                orig_vals = self.ohe.categories_[i][indices]
                reconstructed_cols.append(orig_vals.reshape(-1, 1))
                start += k
                
            X_cat_orig = np.hstack(reconstructed_cols)
        else:
            X_cat_orig = np.zeros((N, 0))
        
        X_final = np.zeros((N, X_num_orig.shape[1] + X_cat_orig.shape[1]))
        if len(self.spec.num_idx) > 0:
            X_final[:, self.spec.num_idx] = X_num_orig
        if len(self.spec.cat_idx) > 0:
            X_final[:, self.spec.cat_idx] = X_cat_orig
            
        return X_final
    
    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
    ) -> ExplanationResult:

        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y_origin).float()
        )
        loader = DataLoader(dataset, batch_size=len(X), shuffle=False)
        
        return self.explain_dataloader(loader)

    def explain_dataloader(
            self,
            dataloader: DataLoader,
            target_class: int = None,
            **kwargs
        ) -> ExplanationResult:
            x_cfs_list = []
            x_origs_list = []
            y_origs_list = []
            y_targets_list = []

            for batch in tqdm(dataloader, desc="Generating CFs"):
                X_batch_np, y_batch_np = batch[0].numpy(), batch[1].numpy()
                B = X_batch_np.shape[0]
                x_cond_tensor = self._transform_to_model_space(X_batch_np)

                if target_class is not None:
                    y_target = torch.full((B,), target_class, device=self.device).long()
                else:
                    y_curr = torch.from_numpy(y_batch_np).long().to(self.device)
                    y_target = 1 - y_curr
                with torch.no_grad():
                    x_cf_model = self.diffusion_model.sample_counterfactual(
                        x_orig=x_cond_tensor, 
                        y_target=y_target
                    )

                x_cf_np = self._inverse_transform_from_model_space(x_cf_model)

                x_cfs_list.append(x_cf_np)
                x_origs_list.append(X_batch_np)
                y_origs_list.append(y_batch_np)
                y_targets_list.append(y_target.cpu().numpy())

            return ExplanationResult(
                x_cfs=np.concatenate(x_cfs_list, axis=0),
                y_cf_targets=np.concatenate(y_targets_list, axis=0),
                x_origs=np.concatenate(x_origs_list, axis=0),
                y_origs=np.concatenate(y_origs_list, axis=0),
                logs={}
            )