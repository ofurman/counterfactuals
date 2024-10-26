import torch
import numpy as np
from sklearn.cluster import KMeans

from counterfactuals.cf_methods.group_ppcef.sparsemax import Sparsemax


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
    def __init__(self, N, D, K, init_from_kmeans=False, X=None):
        super(GCE, self).__init__()
        assert 1 <= K and K <= N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(0 * torch.rand(self.N, 1))
        self.d = torch.nn.Parameter(0 * torch.rand((self.K, self.D)))

        if init_from_kmeans:
            assert X is not None, "X should be provided for KMeans initialization"
            self.s = self._init_from_kmeans(X, K)
        else:
            self.s = torch.nn.Parameter(0.01 * torch.rand(self.N, self.K))
        self.sparsemax = Sparsemax(dim=1)

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

    def loss(self, alpha_s, alpha_k):
        return alpha_s * self.rows_entropy() + alpha_k * self.cols_entropy()

    def get_matrices(self):
        return torch.exp(self.m), self.sparsemax(self.s), self.d
