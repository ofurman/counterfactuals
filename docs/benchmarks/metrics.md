# Evaluation Metrics

Comprehensive metrics for assessing counterfactual quality.

## Validity Metrics

### Coverage

Proportion of instances for which a counterfactual was successfully generated.

$$\text{Coverage} = \frac{|\{x : \text{CF}(x) \neq \emptyset\}|}{|X|}$$

### Validity

Proportion of counterfactuals that achieve the target prediction.

$$\text{Validity} = \frac{|\{x : f(\text{CF}(x)) = y_{\text{target}}\}|}{|X|}$$

## Distance Metrics

### Euclidean Distance (L2)

$$d_{L2}(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}$$

### Manhattan Distance (L1)

$$d_{L1}(x, x') = \sum_{i=1}^{n} |x_i - x'_i|$$

### Mean Absolute Deviation (MAD)

$$d_{MAD}(x, x') = \frac{1}{n} \sum_{i=1}^{n} \frac{|x_i - x'_i|}{\text{MAD}_i}$$

## Sparsity Metrics

### Sparsity

Average number of features changed.

$$\text{Sparsity} = \frac{1}{|X|} \sum_{x \in X} \sum_{i=1}^{n} \mathbb{1}[x_i \neq x'_i]$$

## Plausibility Metrics

### Log-Likelihood Plausibility

Proportion of counterfactuals with log-likelihood above threshold.

$$\text{Plausibility} = \frac{|\{x' : \log p(x') > \tau\}|}{|X|}$$

### Local Outlier Factor (LOF)

Measures how isolated a counterfactual is from training data.

### Isolation Forest Score

Anomaly detection score for counterfactuals.

## Diversity Metrics

### Pairwise Diversity

Average distance between counterfactuals for the same instance.

$$\text{Diversity} = \frac{1}{K(K-1)} \sum_{i \neq j} d(x'_i, x'_j)$$
