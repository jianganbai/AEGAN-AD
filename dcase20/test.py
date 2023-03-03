import torch
import argparse
import yaml
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
import emb_distance as EDIS
from net import Discriminator, AEDC
from datasets import seg_set, clip_set


with open('config.yaml') as fp:
    param = yaml.safe_load(fp)


def get_d_aver_emb(netD, train_set, device):
    netD.eval()
    train_embs = {mid: [] for mid in param['all_mid']}
    with torch.no_grad():
        for idx in range(train_set.get_clip_num()):
            mel, mid, _ = train_set.get_clip_data(idx)
            mel = torch.from_numpy(mel).to(device)
            _, feat_real = netD(mel)
            feat_real = feat_real.squeeze().mean(dim=0).cpu().numpy()
            train_embs[mid].append(feat_real)
    for mid in train_embs.keys():
        train_embs[mid] = np.array(train_embs[mid], dtype=np.float32)
    return train_embs


def test(netD, netG, te_ld, train_embs, device):
    # detect_location, score_type, score_comb= ('x', 'z'), ('2', '1'), ('sum', 'min', 'max')
    D_metric = ['D_maha', 'D_knn', 'D_lof', 'D_cos']
    G_metric = ['G_x_2_sum', 'G_x_2_min', 'G_x_2_max', 'G_x_1_sum', 'G_x_1_min', 'G_x_1_max',
                'G_z_2_sum', 'G_z_2_min', 'G_z_2_max', 'G_z_1_sum', 'G_z_1_min', 'G_z_1_max',
                'G_z_cos_sum', 'G_z_cos_min', 'G_z_cos_max']
    all_metric = D_metric + G_metric
    edetect = EDIS.EmbeddingDetector(train_embs)
    edfunc = {'maha': edetect.maha_score, 'knn': edetect.knn_score,
              'lof': edetect.lof_score, 'cos': edetect.cos_score}
    metric2id = {m: meid for m, meid in zip(all_metric, range(len(all_metric)))}
    id2metric = {v: k for k, v in metric2id.items()}

    def specfunc(x):
        return x.sum(axis=tuple(list(range(1, x.ndim))))
    stfunc = {'2': lambda x, y: (x - y).pow(2),
              '1': lambda x, y: (x - y).abs(),
              'cos': lambda x, y: 1 - F.cosine_similarity(x, y)}
    scfunc = {'sum': lambda x: x.sum().item(),
              'min': lambda x: x.min().item(),
              'max': lambda x: x.max().item()}

    netD.eval()
    netG.eval()
    # {mid: []}
    y_true_all, y_score_all = [{} for _ in metric2id.keys()], [{} for _ in metric2id.keys()]
    with torch.no_grad():
        with tqdm(total=len(te_ld)) as pbar:
            for mel, mid, status in te_ld:  # mel: 1*186*1*128*128
                mel = mel.squeeze(0).to(device)
                _, feat_t = netD(mel)
                recon = netG(mel)
                melz = netG(mel, outz=True)
                reconz = netG(recon, outz=True)
                feat_t = feat_t.mean(axis=0, keepdim=True).cpu().numpy()
                mid, status = mid.item(), status.item()

                for idx, metric in id2metric.items():
                    wn = metric.split('_')[0]
                    if wn == 'D':
                        dname = metric.split('_')[1]
                        score = edfunc[dname](feat_t)

                    elif wn == 'G':
                        dd, st, sc = tuple(metric.split('_')[1:])
                        ori = mel if dd == 'x' else melz
                        hat = recon if dd == 'x' else reconz
                        score = scfunc[sc](specfunc(stfunc[st](hat, ori)))

                    if mid not in y_true_all[idx].keys():
                        y_true_all[idx][mid] = []
                        y_score_all[idx][mid] = []
                    y_true_all[idx][mid].append(status)
                    y_score_all[idx][mid].append(score)
                pbar.update(1)

    aver_of_all_me = []
    for idx in range(len(y_true_all)):
        result = []
        y_true = dict(sorted(y_true_all[idx].items(), key=lambda t: t[0]))  # sort by machine id
        y_score = dict(sorted(y_score_all[idx].items(), key=lambda t: t[0]))
        for mid in y_true.keys():
            AUC_mid = metrics.roc_auc_score(y_true[mid], y_score[mid])
            pAUC_mid = metrics.roc_auc_score(y_true[mid], y_score[mid], max_fpr=param['detect']['p'])
            result.append([AUC_mid, pAUC_mid])
        aver_over_mid = np.mean(result, axis=0)
        aver_of_m = np.mean(aver_over_mid)
        aver_of_all_me.append([aver_over_mid[0], aver_over_mid[1], aver_of_m])
    aver_of_all_me = np.array(aver_of_all_me)
    best_aver = np.max(aver_of_all_me[:, 2])
    best_idx = np.where(aver_of_all_me[:, 2] == best_aver)[0][0]
    best_metric = id2metric[best_idx]

    return aver_of_all_me[best_idx, :], best_metric


def main():
    print('========= Test Machine Type: {} ========='.format(param['mt']['test'][0]))
    device = torch.device('cuda:{}'.format(param['card_id']))
    netD = Discriminator(param)
    netG = AEDC(param)
    pth_file = torch.load(param['model_pth'], map_location=torch.device('cpu'))
    netD.load_state_dict(pth_file['netD'])
    netG.load_state_dict(pth_file['netG'])
    netD.to(device)
    netG.to(device)

    train_data = seg_set(param, param['train_set'], 'train')
    param['all_mid'] = train_data.get_mid()

    print(f"=> Recorded best performance: {pth_file['best_aver']:.4f}")
    for mt in param['mt']['test']:
        mt_test_set = clip_set(param, mt, 'dev', 'test')
        te_ld = DataLoader(mt_test_set,
                           batch_size=1,
                           shuffle=False,
                           num_workers=0)
        train_embs = get_d_aver_emb(netD, train_data, device)

        aver, best_metric = test(netD, netG, te_ld, train_embs, device)
        print(f'{mt} => [AUC: {aver[0]:.4f}] [pAUC: {aver[1]:.4f}] [Average: {aver[2]:.4f}] [Metric: {best_metric}]')


if __name__ == '__main__':
    mt_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    card_num = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt', choices=mt_list, default='valve')
    parser.add_argument('-c', '--card_id', type=int, choices=list(range(card_num)), default=0)
    opt = parser.parse_args()

    param['mt'] = {'train': [opt.mt], 'test': [opt.mt]}
    param['card_id'] = opt.card_id
    param['model_pth'] = utils.get_model_pth(param)

    main()
