# Bag of Tricks for Adversarial Training
Empirical tricks for training state-of-the-art robust models on CIFAR-10

## Statement
The copyrights are reserved by [Tianyu Pang](http://ml.cs.tsinghua.edu.cn/~tianyu/), [Xiao Yang](https://github.com/ShawnXYang) (a summary paper is working in progress).

## Threat Model
We consider the most widely studied setting:
- **L-inf norm constraint with the maximal epsilon be 8/255 on CIFAR-10**.
- **No accessibility to additional data, neither labeled nor unlabeled**.
- **Utilize the min-max framework in [Madry et al. 2018](https://arxiv.org/abs/1706.06083)**.

## Trick Candidates
Importance rate: ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*  ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*  ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*

- **Early stop w.r.t. training epochs** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*).
Early stop w.r.t. training epochs was first implicitly used in [TRADES](https://arxiv.org/abs/1901.08573), while later thoroughly studied by [Rice et al., 2020](https://arxiv.org/abs/2002.11569). The consider this trick as a default choice.

- **Early stop w.r.t. attack iterations** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*). Early stop w.r.t. attack iterations were studied by [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf) and [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). Here we exploit the strategy of the later one, where the authors show that this trick can promote clean accuracy. The relevant flags include `--earlystopPGD` indicates whether apply this trick, while '--earlystopPGDepoch1' and '--earlystopPGDepoch2' separately indicate the epoch to increase the tolerence t by one, as detailed in [Zhang et al. 2020](https://arxiv.org/abs/2002.11242).

- **Warmup w.r.t. learning rate** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). Warmup w.r.t. learning rate was found useful for [FastAT](https://arxiv.org/abs/2001.03994), while [Rice et al., 2020](https://arxiv.org/abs/2002.11569) found that piecewise decay schedule is more compatible with early stop w.r.t. training epochs. The relevant flags include `--warmup_lr` indicates whether apply this trick, while `--warmup_lr_epoch` indicates the end epoch of the gradually increase of learning rate.

- **Warmup w.r.t. epsilon** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). [Qin et al. 2019](https://arxiv.org/abs/1907.02610) use warmup w.r.t. epsilon in their implementation, where the epsilon gradually increase from 0 to 8/255 in the first 15 epochs. Similarly, the relevant flags include `--warmup_eps` indicates whether apply this trick, while `--warmup_eps_epoch` indicates the end epoch of the gradually increase of epsilon.

- **Label smoothing** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). Label smoothing is advocated by [Shafahi et al. 2019](https://arxiv.org/abs/1910.11585) to mimic the adversarial training procedure. The relevant flags include `--labelsmooth` indicates whether apply this trick, while `--labelsmoothvalue` indicates the degree of smoothing applied on the label vectors. When `--labelsmoothvalue=0`, there is no label smoothing applied. 

- **Larger batch size** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). The typical batch size used for CIFAR-10 is 128 in the adversarial setting. In the meanwhile, [Xie et al. 2019](https://arxiv.org/pdf/1812.03411.pdf) apply a large batch size of 4096 to perform adversarial training on ImageNet, where the model is distributed on 128 GPUs and has quite robust performance. The relevant flag is `--batch-size`. According to [Goyal et al. 2017](https://arxiv.org/abs/1706.02677
), we take bs=128 and lr=0.1 as a basis, and scale the lr when we use larger batch size, e.g., bs=256 and lr=0.2.

- **Weight decay** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). The default weight decay used in [Rice et al., 2020](https://arxiv.org/abs/2002.11569) is 5e-4, in [Qin et al. 2019](https://arxiv.org/abs/1907.02610) 2e-4. The relevant flag is `--weight_decay`.

- **Data augmentation** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). [Rice et al., 2020](https://arxiv.org/abs/2002.11569) has tried mixup and cutout. The relevant flags for mixup are `--mixup` and `--mixup-alpha`, while for cutout are `--cutout` and `--cutout-len`. Details can be found in [Rice et al., 2020](https://arxiv.org/abs/2002.11569).

- **Increasing epsilon (ours)** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*). We find that the overfitting of the adversarial training procedure can be explained as: As the model been trained more and more robust, a fixed adversary will become relatively weaker, and consequently the effect of regularization induced by the crafed adversarial examples will also become more insignificant. To overcome this issue, **we need stronger adversaries as the training robust accuracy increase**. We propose to gradually increase epsilon to have a stronger adversary during the training procedure. The relevant flags are '--use_stronger_adv' to indicate whether use this trick, and '--stronger_index' to use a specifc manually designed epsilon schedule. Alternative choice for stronger adversaries include [Multi-target attack](https://arxiv.org/abs/1910.09338).

## Finally Selected Tricks
- **Architecture**: WideResNet-34-10
- **Optimizer**: Momentum SGD with default hyperparameters
- **Total epoch**: `110`
- **Batch size**: `128`
- **Weight decay**: `5e-4`
- **Learning rate**: `lr=0.1`; decay to `lr=0.01` at 100 epoch; decay to `0.001` at 105 epoch
- **Early stop w.r.t. attack iteration**: tolerence `t=1`; let `t=2` at 60 epoch; let `t=3` at 100 epoch
- **Maximal epsilon**: `eps=8/255`; increase to `eps=12/255` at 100 epoch; increase to `eps=16/255` at 105 epoch
- **Attack step size**: `alpha=2/255` 

running command for training:
```python
python train_cifar.py --model WideResNet --attack pgd \
                      --lr-schedule piecewise --norm l_inf --epsilon 8 \
                      --epochs 110 --attack-iters 10 --pgd-alpha 2 \
                      --fname auto \
                      --batch-size 128 \
                      --earlystopPGD --earlystopPGDepoch1 60 --earlystopPGDepoch2 100 \
                      --use_stronger_adv --stronger_index 0
```

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
Our code is based on [Rice et al., 2020](https://arxiv.org/abs/2002.11569) and their github ([here](https://github.com/locuslab/robust_overfitting)).
