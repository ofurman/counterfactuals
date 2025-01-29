import torch
from torch.utils.data import TensorDataset, DataLoader

from counterfactuals.discriminative_models.tabnet import TabNetDiscModel
# Adjust the import based on your actual file structure


def test_tabnet():
    # 1) Create dummy data
    input_dim = 10
    n_samples = 200
    X = torch.randn(n_samples, input_dim)
    # Binary labels [0 or 1]
    y = torch.randint(low=0, high=2, size=(n_samples,))

    # 2) Dataloaders
    dataset = TensorDataset(X, y)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # 3) Instantiate TabNet
    model = TabNetDiscModel(
        input_size=input_dim,
        target_size=1,  # for binary classification
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        device="cpu",
    )

    print("Created TabNet model:\n", model)

    # 4) Train
    model.fit(train_loader, test_loader, epochs=5, lr=0.01, patience=10)
    print("Training finished.")

    # 5) Evaluate
    preds = model.predict(X)  # shape [n_samples]
    proba = model.predict_proba(X)  # shape [n_samples, 2] for binary

    print("Sample predictions:", preds[:10].cpu().numpy())
    print("Sample probabilities:", proba[:3].cpu().numpy())


if __name__ == "__main__":
    test_tabnet()
