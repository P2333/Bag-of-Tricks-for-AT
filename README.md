# Bag-of-Tricks-for-Adversarial-Training
Empirical tricks for training state-of-the-art robust models on CIFAR-10

## Threat Model
We consider the most widely studied setting:
- **L-inf norm constraint with the maximal epsilon be 8/255 on CIFAR-10**.
- **No accessibility to additional data, neither labeled nor unlabeled**.
- **Utilize the min-max framework in [Madry et al. 2018](https://arxiv.org/abs/1706.06083)**.

## Trick Candidates
Importance rate: ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Critical`  ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Useful`  ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `Insignificance`

- **Early stop w.r.t. training epochs** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Critical`) 
- **Early stop w.r.t. attack iterations** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Critical`)
- **Warmup w.r.t. learning rate** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `Insignificance`)
- **Warmup w.r.t. epsilon** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `Insignificance`)
- **Increasing epsilon** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Critical`)
- **Label smoothing** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Useful`)
- **Larger batch size** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `Insignificance`)
- **Weight decay** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Useful`)
- **Data augmentation** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Useful`)
- **Normalization** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `Useful`)

## Empirical Evaluations
*The evaluation results on the baselines are quoted from [Croce et al. 2020](https://arxiv.org/abs/2003.01690) and their github ([here](https://github.com/fra31/auto-attack))*.
The robust accuracy is evaluated at `eps = 8/255`, except for those marked with * for which `eps = 0.031`.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).
|paper           | clean         | APGD-CE | APGD-DLR | FAB | Square | AA  | AA+ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| 89.69| 61.47| 60.64| 60.62| 66.63 | 59.65 | 59.50|
| [(Alayrac et al., 2019)](https://arxiv.org/abs/1905.13725)‡| 86.46| 59.86 | 62.03 |58.20 | 66.37| 56.92| 56.01|
| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| 87.11| 57.00 | 56.96 | 55.40 | 61.99 | 54.99| 54.86|
| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| 85.34| - | - | - | - | 53.60| 53.35|
| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)\*| 84.92| 55.08 | 54.04 | 53.82 | 59.48| 53.18| 53.04|

## Acknowledgement
