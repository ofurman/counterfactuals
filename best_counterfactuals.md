# Mean metrics over top-10 CFs per (dataset, method)

Ranking: CFs ordered by `prox_cont + spars_cat + epsilon_spars` (invalid CFs excluded), top-K selected, then metric means reported.

## bank

| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |
|---|---|---|---|---|---|
| DICE | 0.001 | 0.000 | 0.000 | 0.330 | 0.001 |
| CCHVAE | 0.010 | 0.000 | 0.000 | 0.000 | 0.010 |
| DiCoFlex | 0.016 | 0.000 | 0.000 | 0.253 | 0.016 |

## default

| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |
|---|---|---|---|---|---|
| DICE | 0.003 | 0.000 | 0.036 | 0.213 | 0.039 |
| CCHVAE | 0.005 | 0.000 | 0.000 | 0.000 | 0.005 |
| DiCoFlex | 0.013 | 0.000 | 0.000 | 0.016 | 0.013 |

## adult

| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |
|---|---|---|---|---|---|
| DICE | 0.005 | 0.000 | 0.000 | 0.342 | 0.005 |
| CCHVAE | 0.004 | 0.000 | 0.000 | 0.095 | 0.004 |
| DiCoFlex | 0.012 | 0.000 | 0.000 | 0.154 | 0.012 |

## gmc

| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |
|---|---|---|---|---|---|
| DICE | 0.001 | 0.000 | 0.000 | 0.151 | 0.001 |
| CCHVAE | 0.005 | 0.000 | 0.000 | 0.000 | 0.005 |
| DiCoFlex | 0.004 | 0.000 | 0.000 | 0.091 | 0.004 |

## lending-club

| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |
|---|---|---|---|---|---|
| DICE | 0.002 | 0.000 | 0.000 | 0.248 | 0.002 |
| CCHVAE | 0.010 | 0.000 | 0.000 | 0.001 | 0.010 |
| DiCoFlex | 0.016 | 0.000 | 0.000 | 0.126 | 0.016 |

