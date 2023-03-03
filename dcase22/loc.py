import os
import yaml
import torch
import numpy as np
import argparse
import librosa.display
from tqdm import tqdm
from matplotlib import pyplot as plt

import utils
import datasets
from net import Generator

with open('./config.yaml') as fp:
    param = yaml.safe_load(fp)


def loc():
    train_set = datasets.train_dataset(param)
    test_set = datasets.test_dataset(param, 'dev', 'test')
    test_clip_name = test_set.get_clip_name()

    # calculate the average spectrogram of the dev set
    cindex = []
    for cid in range(train_set.all_attri.shape[0]):
        if train_set.all_attri[cid, 0] in [0, 1, 2]:
            cindex.append(cid)
    aver = np.mean(train_set.all_clip_spec[cindex, :, :], axis=0)
    aver = (aver + 1) / 2

    # set up network
    device = torch.device(f"cuda:{param['card_id']}")
    AE = Generator(param)
    pth_file = torch.load(utils.get_model_pth(param), map_location=torch.device('cpu'))
    pretrained_dict = pth_file['netG']
    AE.load_state_dict(pretrained_dict)
    AE.to(device)
    AE.eval()

    def query_denoise(test_set, cid):
        with torch.no_grad():
            mel, _, _ = test_set.__getitem__(cid)
            mel = torch.from_numpy(mel).to(device)
            recon = AE(mel).squeeze().cpu().numpy()
            recon_all = np.zeros(test_set.all_clip_spec.shape[1:], dtype=np.float32)
            for col in range(recon_all.shape[1]):  # 0-312
                leftcol = max(0, col - param['feat']['frame_num'] + 1)
                rightcol = min(col + 1, mel.shape[0])
                col_range = list(range(leftcol, rightcol))
                aver_col = np.zeros(param['feat']['mel_bin'])
                for ccol in col_range:
                    aver_col += recon[ccol, :, col - ccol]
                aver_col /= len(col_range)
                recon_all[:, col] = aver_col
        return recon_all

    with tqdm(total=test_set.all_clip_spec.shape[0], desc='drawing maps') as pbar:
        for cid in range(test_set.all_clip_spec.shape[0]):
            ori = query_denoise(test_set, cid)
            ori = (ori + 1) / 2
            delta = np.abs(ori - aver)

            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 6))
            librosa.display.specshow(ori, ax=ax[0], sr=16000, vmin=0, vmax=1)
            librosa.display.specshow(aver, ax=ax[1], sr=16000, vmin=0, vmax=1)
            librosa.display.specshow(delta, ax=ax[2], sr=16000)

            name = test_clip_name[cid]
            sec = int(name.split('_')[1][1])
            dom = name.split('_')[2]
            true = 1 if name.split('_')[4] == 'anomaly' else 0
            fname = f"sec{sec}_{dom}_true{true}_{cid}.jpg"
            save_file = os.path.join(f'./ano_loc/{opt.mt}/', fname)
            fig.savefig(save_file, bbox_inches='tight')
            plt.close()
            pbar.update(1)


if __name__ == '__main__':
    mt_list = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt', choices=mt_list, default='ToyCar')
    parser.add_argument('-c', '--card', type=int, choices=list(range(8)), default=3)
    opt = parser.parse_args()

    param['mt'] = opt.mt
    param['card_id'] = opt.card

    os.makedirs('./ano_loc/', exist_ok=True)
    os.makedirs(f'./ano_loc/{opt.mt}/', exist_ok=True)
    if len(os.listdir(f'./ano_loc/{opt.mt}/')) > 0:
        os.remove(f'rm ./ano_loc/{opt.mt}/*.jpg')

    loc()
