import numpy as np
import torch


def generate_multiclass_counterfactuals(
    model,
    factual_points: np.ndarray,
    target_class: int,
    p_value: float,
    mask: np.ndarray,
    n_samples: int = 10,
    temperature: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes: int = None
):
    """
    Generate counterfactual samples for given factual points targeting a specific class.

    Args:
        model: Trained flow model
        factual_points: Array of factual points to generate counterfactuals for
        target_class: Target class to generate counterfactuals for
        p_value: p-norm sparsity
        mask: Immutable features mask
        n_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diverse)
        device: Device to use for generation
        num_classes: Number of classes in the dataset

    Returns:
        Array of generated counterfactual samples of shape (factual_points.shape[0], n_samples, factual_points.shape[1])
    """
    model.eval()
    all_counterfactuals = np.zeros((factual_points.shape[0], n_samples, factual_points.shape[1]))
    all_log_probs = np.zeros((factual_points.shape[0], n_samples))
    batch_size = 256

    p = torch.tensor([p_value], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for factual_idx in range(0, len(factual_points), batch_size):
            end_idx = factual_idx + batch_size
            factual = factual_points[factual_idx:end_idx]
            cur_batch_size = len(factual)

            # Convert to tensor and add batch dimension
            factual_tensor = torch.tensor(factual, dtype=torch.float32).to(device)

            # Create a one-hot encoding for the target class
            class_one_hot = np.zeros(num_classes)
            class_one_hot[target_class] = 1
            class_tensor = torch.tensor(class_one_hot, dtype=torch.float32).unsqueeze(0).to(device)
            class_tensor = class_tensor.repeat((cur_batch_size, 1))
            p_tensor = p.repeat((cur_batch_size, 1))
            mask_tensor = mask.repeat((cur_batch_size, 1))

            # Combine factual point, class one-hot encoding, feature mask and p-norm
            context = torch.cat([factual_tensor, class_tensor, mask_tensor, p_tensor], dim=1)

            # Generate samples
            samples, log_probs = model.sample_and_log_prob(
                num_samples=n_samples,
                context=context,
                temp=temperature
            )

            log_probs = log_probs.squeeze(0).cpu().numpy()
            samples = samples.squeeze(0)

            # Add to results
            all_counterfactuals[factual_idx:end_idx] = samples.cpu().numpy()
            all_log_probs[factual_idx:end_idx] = log_probs

    return all_counterfactuals, all_log_probs
