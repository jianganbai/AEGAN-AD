# AEGAN-AD
Official pytorch implementation of AEGAN-AD.

<img src="./framework.jpg">

## Introduction

Automatic detection of machine anomaly remains challenging for machine learning. We believe the capability of generative adversarial network (GAN) suits the need of machine audio anomaly detection, yet rarely has this been investigated by previous work. In this paper, we propose AEGAN-AD, a totally unsupervised approach in which the generator (also an autoencoder) is trained to reconstruct input spectrograms. It is pointed out that the denoising nature of reconstruction deprecates its capacity. Thus, the discriminator is redesigned to aid the generator during both training stage and detection stage. The performance of AEGAN-AD on the dataset of DCASE 2022 Challenge TASK 2 demonstrates the state-of-the-art result on five machine types. A novel anomaly localization method is also investigated.

## Preparation

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

## Training

Hyper parameters are stored in `config.yaml`.

To train a model, please enter:
```
    python train.py --mt {machine type} -c {card_id} --seed {seed}
```

To test a model, please enter:
```
    python test.py --mt {machine type} -d -c {card_id}
```

## Pretrained Dicts

Pretrained dicts are provided via [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/3d4ddf315bcf41078e07/).

Place the dict under `./model` (or modify in `config.yaml` to your custom path) and rename the dict as "{machine type}.pth".

Substitute `config.yaml` by the corresponding config file in `./pretrain/`.

To verify the performance, please enter:
```
    python test.py --mt {machine type} -d -c {card_id}
```

## Result

|        | bearing   | fan         | gearbox     | slider | ToyCar    |
| ------ | --------- | ----------- | ----------- | ------ | --------- |
| hmean  | 76.03     | 65.83       | 75.27       | 74.06  | 78.46     |
| metric | G_z_1_sum | G_z_cos_min | G_z_cos_min | D_LOF  | G_z_1_sum |

- G_z means using the reconstruction error in the latent space of the generator. 1 stands for L1 norm.
- D_LOF means using the LOF algorithm on the embedding of the discriminator.
