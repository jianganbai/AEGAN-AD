# AEGAN-AD
Official pytorch implementation of AEGAN-AD.

<img src="./framework.jpg">

## Introduction

Automatic detection of machine anomaly remains challenging for machine learning. We believe the capability of generative adversarial network (GAN) suits the need of machine audio anomaly detection, yet rarely has this been investigated by previous work. In this paper, we propose AEGAN-AD, a totally unsupervised approach in which the generator (also an autoencoder) is trained to reconstruct input spectrograms. It is pointed out that the denoising nature of reconstruction deprecates its capacity. Thus, the discriminator is redesigned to aid the generator during both training stage and detection stage. The performance of AEGAN-AD on the dataset of DCASE 2022 Challenge TASK 2 demonstrates the state-of-the-art result on five machine types. A novel anomaly localization method is also investigated.



Let $\mathbb{P}_r$ and $\mathbb{P}_g$ denote the real distribution and the generated distribution respectively. The discriminator is trained to minimize the following function:

$$\mathcal{L}_D=\mathop{\mathbb{E}}\limits_{\tilde{x}\sim{\mathbb{P}_g}}\left[D(\tilde{x})\right]-\mathop{\mathbb{E}}\limits_{x\sim\mathbb{P}_r}\left[D(x)\right]+{\lambda}\mathop{E}_{\hat{x}\sim\mathbb{P}_{\hat{x}}}\left[\left(\Vert\nabla_{\hat{x}}D(\hat{x})\Vert_2-1\right)^2\right]$$

where $\lambda$ is weight of gradient penalty. $\mathbb{P}_{\hat{x}}$ is a distribution of samples with the following linear combination of that drawn from $\mathbb{P}_r$ and $\mathbb{P}_g$:

  $$\hat{x}=\alpha{\cdot}x+(1-\alpha){\cdot}{\tilde{x}}$$

where $x{\sim}\mathbb{P}_r$ and $\tilde{x}{\sim}\mathbb{P}_g$. $\alpha$ is a randomly selected parameter.



An alternative reconstruction-based loss function is adopted for the generator. Let $f(\cdot)$ denote the embedding of the discriminator. The loss function of the generator is formulated as follows:

$$\mathcal{L}_G=\mathop{\mathbb{E}}\limits_{x{\sim}\mathbb{P}_r}\left[\Vert{x-G(x)}\Vert_2^2\right] +{\mu_1}{\Vert}\mathop{\mathbb{E}}\limits_{x{\sim}\mathbb{P}_r}\left[f(x)\right]-\mathop{\mathbb{E}}_{\tilde{x}\sim\mathbb{P}_g}\left[f(\tilde{x})\right]{\Vert}_2^2$$

where the first term is the norm of the reconstruction error and the second term is the feature matching loss. A modified $\mathcal{L}_G$ is proposed for some machine types, which measures the feature matching loss via both mean and standard deviation:

$$\mathcal{L}_G=\mathop{\mathbb{E}}\limits_{x{\sim}\mathbb{P}_r}\left[\Vert{x-G(x)}\Vert_2^2\right]+{\mu_1}{\Vert}\mathop{\mathbb{E}}\limits_{x{\sim}\mathbb{P}_r}\left[f(x)\right]-\mathop{\mathbb{E}}_{\tilde{x}\sim\mathbb{P}_g}\left[f(\tilde{x})\right]{\Vert}_2^2+{\mu_2}{\Vert}\mathop{\sigma}\limits_{x{\sim}\mathbb{P}_r}\left[f(x)\right]-\mathop{\sigma}_{\tilde{x}\sim\mathbb{P}_g}\left[f(\tilde{x})\right]{\Vert}_2^2$$

where $\sigma[f(x)]$ is a vector, of which each element is the standard deviation of the corresponding dimension of the embedding.



Experiments were conducted on DCASE 20 datset and DCASE 22 dataset. Code for each dataset can be found in the corresponding directory. Please refer to the corresponding instructions below. Two sets of code are mainly the same.

## DCASE 20

### Preparation

Download the dataset from [DCASE official website](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds).

The directory should be organized in the following manner:

```
dataset_dir/
    dev_data/
        fan/
            train/
            test/
            ...
    eval_data/
        fan/
            train/
            test/
            ...
```

Then clone this repository. Modify the `dataset_dir` term in `config.yaml` to your dataset path.

Finally, install required packages:

```
    pip install -r requirements
```

### Training

Hyper parameters are stored in `config.yaml`.

To train a model, please enter:

```
    python train.py --mt {machine type} -c {card_id} --seed {seed}
```

To test a model, please enter:

```
    python test.py --mt {machine type} -c {card_id}
```

### Pretrained Dicts

Pretrained dicts are provided via [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/3d4ddf315bcf41078e07/).

Place the dict under `./model` (or modify in `config.yaml` to your custom path).

Substitute `config.yaml` by the corresponding config file in `./pretrain/`.

To verify the performance, please enter:

```
    python test.py --mt {machine type} -c {card_id}
```

### Result

|         | fan   | pump  | slider      | ToyCar | ToyConveyor | valve        |
| ------- | ----- | ----- | ----------- | ------ | ----------- | ------------ |
| average | 77.01 | 81.26 | 86.50       | 86.62  | 73.27       | 77.60        |
| metric  | D-LOF | D-LOF | G-x-L2-mean | D-KNN  | G-z-L2-min  | G-z-cos-mean |

- G-z means using the reconstruction error in the latent space of the generator. L2 stands for L2 norm.
- D-LOF means using the LOF algorithm on the embedding of the discriminator.

## DCASE 22

### Preparation

Download the dataset from [DCASE official website](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).

The directory should be organized in the following manner:

```
dataset_dir/
    dev_data/
        bearing/
            train/
            test/
            ...
    eval_data/
        bearing/
            train/
            test/
            ...
```

Then clone this repository. Modify the `dataset_dir` term in `config.yaml` to your dataset path.

Finally, install required packages:
```
    pip install -r requirements
```

### Training

Hyper parameters are stored in `config.yaml`.

To train a model, please enter:
```
    python train.py --mt {machine type} -c {card_id} --seed {seed}
```

To test a model, please enter:
```
    python test.py --mt {machine type} -d -c {card_id}
```

### Pretrained Dicts

Pretrained dicts are provided via [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/3d4ddf315bcf41078e07/).

Place the dict under `./model` (or modify in `config.yaml` to your custom path).

Substitute `config.yaml` by the corresponding config file in `./pretrain/`.

To verify the performance, please enter:
```
    python test.py --mt {machine type} -d -c {card_id}
```

### Result

|        | bearing    | fan         | gearbox     | slider | ToyCar    |
| ------ | ---------- | ----------- | ----------- | ------ | --------- |
| hmean  | 76.03      | 65.83       | 75.27       | 74.06  | 78.46     |
| metric | G-z-L1-sum | G-z-cos-min | G-z-cos-min | D-LOF  | G-z-1-sum |

- G-z means using the reconstruction error in the latent space of the generator. 1 stands for L1 norm.
- D-LOF means using the LOF algorithm on the embedding of the discriminator.
