# Bag-of-Tricks-for-Adversarial-Training
Empirical tricks for training state-of-the-art robust models on CIFAR-10

## Statement
The copyrights are reserved by Tianyu Pang, Xiao Yang

## Threat Model
We consider the most widely studied setting:
- **L-inf norm constraint with the maximal epsilon be 8/255 on CIFAR-10**.
- **No accessibility to additional data, neither labeled nor unlabeled**.
- **Utilize the min-max framework in [Madry et al. 2018](https://arxiv.org/abs/1706.06083)**.

## Trick Candidates
Importance rate: ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*  ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*  ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*

- **Early stop w.r.t. training epochs** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*).
Early stop w.r.t. training epochs was first implicitly used in [TRADES](https://arxiv.org/abs/1901.08573), while later thoroughly studied by [Rice et al., 2020](https://arxiv.org/abs/2002.11569).

- **Early stop w.r.t. attack iterations** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*). Early stop w.r.t. attack iterations were studied by [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf) and [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). Here we exploit the strategy of the later one.

- **Warmup w.r.t. learning rate** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). Warmup w.r.t. learning rate was found useful for [FastAT](https://arxiv.org/abs/2001.03994), while [Rice et al., 2020](https://arxiv.org/abs/2002.11569) found that piecewise decay schedule is more compatible with early stop w.r.t. training epochs.

- **Warmup w.r.t. epsilon** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). [Qin et al. 2019](https://arxiv.org/abs/1907.02610) use warmup w.r.t. epsilon in their implementation, where the epsilon gradually increase from 0 to 8/255 in the first 15 epochs.

- **Label smoothing** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). Label smoothing is advocated by [Shafahi et al. 2019](https://arxiv.org/abs/1910.11585) to mimic the adversarial training procedure.

- **Larger batch size** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). The typical batch size used for CIFAR-10 is 128 in the adversarial setting. In the meanwhile, [Xie et al. 2019](https://arxiv.org/pdf/1812.03411.pdf) apply a large batch size of 4096 to perform adversarial training on ImageNet, where the model is distributed on 128 GPUs and has quite robust performance.

- **Weight decay** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). The default weight decay used in [Rice et al., 2020](https://arxiv.org/abs/2002.11569) is 5e-4, in [Qin et al. 2019](https://arxiv.org/abs/1907.02610) 2e-4.

- **Data augmentation** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). [Rice et al., 2020](https://arxiv.org/abs/2002.11569) has tried mixup and cutout.

- **Normalization** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*).

- **Increasing epsilon** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*).

## Finally Selected Tricks
- Architecture: WideResNet-34-10
- Optimizer: Momentum SGD with default hyperparameters
- Total epoch: `110`
- Batch size: `128`
- Weight decay: `5e-4`
- Learning rate: `lr=0.1`; decay to `lr=0.01` at 100 epoch; decay to `0.001` at 105 epoch
- Early stop w.r.t. attack iteration: tolerence `t=1`; let `t=2` at 60 epoch; let `t=3` at 100 epoch
- Maximal epsilon: `eps=8/255`; increase to `eps=12/255` at 100 epoch; increase to `eps=16/255` at 105 epoch
- Attack step size: `alpha=2/255` 

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
