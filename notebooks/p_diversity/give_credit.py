import time

import numpy as np
import torch
from counterfactuals.datasets import GiveMeSomeCreditDataset

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = GiveMeSomeCreditDataset(
    train_file="data/GiveMeSomeCredit-training.csv",
    test_file="data/GiveMeSomeCredit-testing.csv"
)

print(f"Train shape: {dataset.X_train.shape}, Test shape: {dataset.X_test.shape}")

from counterfactuals.generative_models import MaskedAutoregressiveFlow as baseMAF, MaskedAutoregressiveFlow

flow_train_dataloader = dataset.train_dataloader(
    batch_size=128, shuffle=True, noise_lvl=0.03
)
flow_test_dataloader = dataset.test_dataloader(batch_size=128, shuffle=False)
global_start_time = time.time()
# Move model to CUDA
flow = baseMAF(
    features=len(dataset.feature_columns),
    hidden_features=16,
    num_blocks_per_layer=4,
    num_layers=8,
    context_features=1,
    device=device
).to(device)

flow.fit(flow_train_dataloader, flow_test_dataloader, num_epochs=1, patience=50)

from counterfactuals.discriminative_models import MultilayerPerceptron

# Move model to CUDA
disc_model = MultilayerPerceptron(
    input_size=len(dataset.feature_columns),
    hidden_layer_sizes=[256, 256],
    target_size=1,
    dropout=0.2,
    device=device
).to(device)

train_dataloader = dataset.train_dataloader(batch_size=64, shuffle=True, noise_lvl=0.0)
test_dataloader = dataset.test_dataloader(batch_size=64, shuffle=False)

disc_model.fit(train_dataloader, test_dataloader, epochs=100, patience=100, lr=1e-3)

# Move data to CUDA before prediction
dataset.X_test = torch.tensor(dataset.X_test, device=device, dtype=torch.float32)  # Ensure data is on the same device
y_pred = (disc_model.predict(dataset.X_test).detach().cpu().numpy() > 0.5).astype(int)
y_true = dataset.y_test
print(f"Accuracy: {np.mean(y_pred == y_true)}")

from counterfactuals.metrics import CFMetrics

def l0_distance(X_cf, X_test):
    return np.sum(X_cf != X_test, axis=1).mean()

cfs = []
start_time = time.time()
with torch.no_grad():
    for x in dataset.X_test:
        # Move data to CUDA before inference
        x_tensor = x.unsqueeze(0) if x.ndimension() == 1 else x  # Ensure correct tensor shape
        x_tensor = x_tensor.to(device)  # Move to same device
        points, log_prob = flow.sample_and_log_prob(100, context=x_tensor)
        cfs.append(points.cpu())
generation_time = time.time() - start_time
print(f"Time to generate all counterfactuals: {generation_time:.4f} seconds")

# cfs = torch.stack(cfs).squeeze().permute(1, 0, 2).numpy()

# Second Flow Training for Counterfactual Generation
cf = MaskedAutoregressiveFlow(
    features=dataset.X_test.shape[1],
    hidden_features=16,
    num_blocks_per_layer=2,
    num_layers=2,
    context_features=dataset.X_test.shape[1],
    device=device
).to(device)

cf.fit(
    train_dataloader, test_dataloader, num_epochs=1, learning_rate=1e-3, patience=100, lambda_dist=0.2, checkpoint_path="best_cf_model_dist_0.2_give_credit.pt"
)

end_time = time.time() - global_start_time
print(f"Time to finish all: {end_time:.4f} seconds")

cf.load("best_cf_model_dist_0.2_give_credit.pt")

all_metrics = []
for i in range(cfs.shape[0]):
    metrics = CFMetrics(
        X_cf=cfs[i],
        y_target=np.abs(dataset.y_test - 1),
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test.cpu().numpy(),  # Ensure consistency with numpy operations
        y_test=dataset.y_test,
        gen_model=cf,
        disc_model=disc_model,
        continuous_features=dataset.numerical_features,
        categorical_features=[],
        prob_plausibility_threshold=1.2,
    )
    metric_results = metrics.calc_all_metrics()
    metric_results["L0 Distance"] = l0_distance(cfs[i], dataset.X_test.cpu().numpy())
    all_metrics.append(metric_results)

mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}

for key in mean_metrics:
    print(f"{key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
