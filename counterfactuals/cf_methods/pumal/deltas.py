import torch
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple, Optional, Any

from counterfactuals.cf_methods.pumal.sparsemax import Sparsemax


class GradStrategy(Enum):
    ZERO = auto()            # Zero out gradient (no change)
    INCREASE_ONLY = auto()   # Only allow positive gradients (increase only)
    DECREASE_ONLY = auto()   # Only allow negative gradients (decrease only)
    UNRESTRICTED = auto()    # No gradient restriction


@dataclass
class DimConfig:
    """Configuration for a single dimension in the optimization process."""
    strategy: GradStrategy = GradStrategy.UNRESTRICTED
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    
    def has_range(self) -> bool:
        """Check if this dimension has a valid clamping range defined."""
        return self.min_val is not None and self.max_val is not None


class PPCEF_2(torch.nn.Module):
    def __init__(self, N, D, K):
        super(PPCEF_2, self).__init__()
        assert K == N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.d = torch.nn.Parameter(torch.zeros((N, D)))

    def forward(self):
        return self.d

    def get_matrices(self):
        return torch.ones(self.N, 1), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class ARES(torch.nn.Module):
    def __init__(self, N, D, K=1):
        super(ARES, self).__init__()
        assert K == 1, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.d = torch.nn.Parameter(torch.zeros(self.K, self.D))

    def forward(self):
        return torch.ones(self.N, self.K) @ self.d

    def get_matrices(self):
        return torch.ones(self.N, 1), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class GLOBAL_CE(torch.nn.Module):
    def __init__(self, N, D, K):
        super(GLOBAL_CE, self).__init__()
        assert K == 1, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(torch.zeros(self.N, 1))
        self.d = torch.nn.Parameter(torch.zeros((self.K, self.D)))

    def forward(self):
        return torch.exp(self.m) @ self.d

    def get_matrices(self):
        return torch.exp(self.m), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class GCE(torch.nn.Module):
    def __init__(self, N, D, K, init_from_kmeans=False, X=None, 
                 dim_configs: Optional[Dict[int, DimConfig]] = None):
        """
        Initialize GCE with per-dimension gradient strategies and value ranges.
        
        Args:
            N (int): Number of instances
            D (int): Dimensionality of the data
            K (int): Number of clusters
            init_from_kmeans (bool): Whether to initialize S from KMeans clustering
            X (tensor): Data for KMeans initialization
            dim_configs (Dict[int, DimConfig]): Dictionary mapping dimension indices to 
                                                DimConfig objects specifying gradient behavior
                                                and clamping ranges
        """
        super(GCE, self).__init__()
        assert 1 <= K and K <= N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(0 * torch.rand(self.N, 1))
        self.d = torch.nn.Parameter(0 * torch.rand((self.K, self.D)))
        
        # Initialize dimension configurations
        self.dim_configs = dim_configs if dim_configs is not None else {}
        
        # Register hooks if needed
        if self.dim_configs:
            self.d.register_hook(self._gradient_hook)
            self._register_clamp_hook()

        if init_from_kmeans:
            assert X is not None, "X should be provided for KMeans initialization"
            self.s = self._init_from_kmeans(X, K)
        else:
            self.s = torch.nn.Parameter(0.01 * torch.rand(self.N, self.K))
        self.sparsemax = Sparsemax(dim=1)
        
    def _register_clamp_hook(self):
        """
        Register a hook that clamps values to specified ranges after each optimization step.
        """
        def clamp_hook(module, input, output):
            with torch.no_grad():
                for dim, config in self.dim_configs.items():
                    if config.has_range():
                        self.d.data[:, dim].clamp_(config.min_val, config.max_val)
        
        self.register_forward_hook(clamp_hook)
        
    def clamp_parameters(self):
        """
        Manually clamp parameters to their specified ranges.
        This can be called explicitly if needed.
        """
        with torch.no_grad():
            for dim, config in self.dim_configs.items():
                if config.has_range():
                    self.d.data[:, dim].clamp_(config.min_val, config.max_val)
                
    def _gradient_hook(self, grad):
        """
        Hook to modify gradients for specified dimensions based on their strategies.
        
        This hook applies different gradient strategies to different dimensions based on
        the GradStrategy enum in each dimension's DimConfig.
        """
        # Create a copy of the gradient
        modified_grad = grad.clone()
        
        # Apply strategies to each dimension
        for dim, config in self.dim_configs.items():
            if config.strategy == GradStrategy.ZERO:
                modified_grad[:, dim] = 0
            elif config.strategy == GradStrategy.INCREASE_ONLY:
                # Zero out negative gradients
                mask = (modified_grad[:, dim] < 0)
                if mask.any():
                    modified_grad[:, dim][mask] = 0
            elif config.strategy == GradStrategy.DECREASE_ONLY:
                # Zero out positive gradients
                mask = (modified_grad[:, dim] > 0)
                if mask.any():
                    modified_grad[:, dim][mask] = 0
                    
        return modified_grad

    def determinant_diversity_penalty(self, vectors):
        """
        Computes the determinant-based diversity penalty for a set of vectors.
        Args:
            vectors (torch.Tensor): Tensor of shape [6, 23] where each row is a vector.

        Returns:
            torch.Tensor: Penalty term encouraging diversity.
        """
        # Compute Gram matrix: G = V @ V.T
        gram_matrix = torch.mm(vectors, vectors.T)  # Shape: [6, 6]

        # Add small regularization for numerical stability (ensure positive semi-definite)
        epsilon = 1e-5
        gram_matrix_regularized = gram_matrix + epsilon * torch.eye(
            gram_matrix.size(0)
        ).to(vectors.device)

        # Compute the log-determinant of the Gram matrix
        log_det = torch.logdet(gram_matrix_regularized)

        # The penalty term: Negative log-determinant (we want to maximize det)
        penalty = -log_det

        return penalty

    def _init_from_kmeans(self, X, K):
        kmeans = KMeans(n_clusters=K, random_state=42).fit(X)
        group_labels = kmeans.labels_
        group_labels_one_hot = np.zeros((group_labels.size, group_labels.max() + 1))
        group_labels_one_hot[np.arange(group_labels.size), group_labels] = 1
        assert group_labels_one_hot.shape[1] == K
        assert group_labels_one_hot.shape[0] == X.shape[0]
        return torch.from_numpy(group_labels_one_hot).float()

    def _entropy_loss(self, prob_dist):
        prob_dist = torch.clamp(prob_dist, min=1e-9)
        row_wise_entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=1)
        return row_wise_entropy

    def forward(self):
        return torch.exp(self.m) * self.sparsemax(self.s) @ self.d

    def rows_entropy(self):
        row_wise_entropy = self._entropy_loss(self.sparsemax(self.s))
        return row_wise_entropy

    def cols_entropy(self):
        s_col_prob = self.sparsemax(self.s).sum(axis=0) / self.sparsemax(self.s).sum()
        s_col_prob = s_col_prob.clamp(min=1e-9)
        col_wise_entropy = -torch.sum(s_col_prob * torch.log(s_col_prob))
        return col_wise_entropy

    def loss(self, alpha_s, alpha_k, alpha_d):
        # return alpha_s * self.rows_entropy() + alpha_s * torch.norm(self.d, p=0, dim=1).sum() # + alpha_k * self.cols_entropy()
        return (
            alpha_s * self.rows_entropy()
            + alpha_d * torch.relu(self.determinant_diversity_penalty(self.d))
            + alpha_k * self.cols_entropy()
        )

    def get_matrices(self):
        return torch.exp(self.m), self.sparsemax(self.s), self.d
