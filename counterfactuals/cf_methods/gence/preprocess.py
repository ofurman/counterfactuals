import torch
from torch.utils.data import Dataset


class PairDistanceDataset(Dataset):
    def __init__(self, class_zero, class_one, length=None):
        """
        Initialize with two arrays, one for each class.
        """
        self.length = length
        self.class_zero = torch.tensor(class_zero, dtype=torch.float32)
        self.class_one = torch.tensor(class_one, dtype=torch.float32)

        # Calculate pairwise distances between zero and one classes
        self.zero_one_distance = torch.cdist(self.class_zero, self.class_one) ** 4

        self.size_zero = class_zero.shape[0]
        self.size_one = class_one.shape[0]

    def __len__(self):
        # The total combinations are len(class_zero) * len(class_one)
        if self.length is not None:
            return self.length
        return self.size_zero * self.size_one

    def get_specific_item(self, idx):
        """
        Get the specific item based on the index.
        Sample the second point based on the distance weight from another class.
        """
        if idx < self.size_zero:
            i = idx
            x_orig = self.class_zero[i]

            # Calculate weights for sampling y
            zero_one_weight = 1 / self.zero_one_distance[i]
            zero_one_weight /= zero_one_weight.sum()

            j = torch.multinomial(zero_one_weight, num_samples=1).item()
            x_cf = self.class_one[j]
        else:
            i = idx + self.size_zero
            x_orig = self.class_one[i]

            # Calculate weights for sampling y
            zero_one_weight = 1 / self.zero_one_distance[:, i]
            zero_one_weight /= zero_one_weight.sum()

            j = torch.multinomial(zero_one_weight, num_samples=1).item()
            x_cf = self.class_zero[j]
        return torch.tensor(x_cf, dtype=torch.float32), torch.tensor(
            x_orig, dtype=torch.float32
        )

    def __getitem__(self, idx):
        """
        Randomly select a point from one class.
        Sample the second point based on the distance weight from another class.
        """
        if torch.rand(1) > 0.5:
            i = torch.randint(0, self.size_zero, (1,)).item()
            x_orig = self.class_zero[i]

            # Calculate weights for sampling y
            zero_one_weight = 1 / self.zero_one_distance[i]
            zero_one_weight /= zero_one_weight.sum()

            j = torch.multinomial(zero_one_weight, num_samples=1).item()
            x_cf = self.class_one[j]
        else:
            i = torch.randint(0, self.size_one, (1,)).item()
            x_orig = self.class_one[i]

            # Calculate weights for sampling y
            zero_one_weight = 1 / self.zero_one_distance[:, i]
            zero_one_weight /= zero_one_weight.sum()

            j = torch.multinomial(zero_one_weight, num_samples=1).item()
            x_cf = self.class_zero[j]
        return torch.tensor(x_cf, dtype=torch.float32), torch.tensor(
            x_orig, dtype=torch.float32
        )
