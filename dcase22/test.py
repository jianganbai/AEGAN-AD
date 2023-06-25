import torch
import argparse
import yaml
import numpy as np
import os
import scipy
import sys
from tqdm import tqdm
from sklearn import metrics
import csv
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
import emb_distance as EDIS
from net import Discriminator, Generator
from datasets import train_dataset, test_dataset


with open('config.yaml') as fp:
    param = yaml.safe_load(fp)


def get_d_aver_emb(netD, train_set, device):
    netD.eval()
    train_embs = {sec: {'source': [], 'target': []} for sec in param['all_sec']}
    with torch.no_grad():
        for idx in range(train_set.get_clip_num()):
            mel, attri, _ = train_set.get_clip_data(idx)
            mel = torch.from_numpy(mel).to(device)
            _, feat_real = netD(mel)
            feat_real = feat_real.squeeze().mean(dim=0).cpu().numpy()
            dom = 'source' if attri[0, 1] == 0 else 'target'
            train_embs[attri[0, 0]][dom].append(feat_real)
    for sec in train_embs.keys():
        for dom in ['source', 'target']:
            train_embs[sec][dom] = np.array(train_embs[sec][dom], dtype=np.float32)
    return train_embs


# @profile
def test(netD, netG, te_ld, file_list, train_embs,
         set_sec, device, data_type, score_metric=None, cal_auc=True):
    # detect_domain, score_type, score_comb= ('x', 'z'), ('2', '1'), ('sum', 'min', 'max')
    D_metric = ['D_maha', 'D_knn', 'D_lof', 'D_cos']
    G_metric = ['G_x_2_sum', 'G_x_2_min', 'G_x_2_max', 'G_x_1_sum', 'G_x_1_min', 'G_x_1_max',
                'G_z_2_sum', 'G_z_2_min', 'G_z_2_max', 'G_z_1_sum', 'G_z_1_min', 'G_z_1_max',
                'G_z_cos_sum', 'G_z_cos_min', 'G_z_cos_max']
    all_metric = D_metric + G_metric
    metric2id = {m: mid for m, mid in zip(all_metric, range(len(all_metric)))}
    id2metric = {v: k for k, v in metric2id.items()}

    def specfunc(x):
        return x.sum(axis=tuple(list(range(1, x.ndim))))
    stfunc = {'2': lambda x, y: (x - y).pow(2),
              '1': lambda x, y: (x - y).abs(),
              'cos': lambda x, y: 1 - F.cosine_similarity(x, y)}
    scfunc = {'sum': lambda x: x.sum().item(),
              'min': lambda x: x.min().item(),
              'max': lambda x: x.max().item()}
    edetect = EDIS.EmbeddingDetector(train_embs)
    edfunc = {'maha': edetect.maha_score, 'knn': edetect.knn_score,
              'lof': edetect.lof_score, 'cos': edetect.cos_score}

    netG.eval()
    # {sec: {'source': [], 'target': []}}
    y_true_all, y_score_all = [{} for _ in metric2id.keys()], [{} for _ in metric2id.keys()]  # for calculating AUC
    file_sec_list, score_sec_list = [{} for _ in metric2id.keys()], [{} for _ in metric2id.keys()]  # for recording score
    with torch.no_grad():
        with tqdm(total=len(file_list)) as pbar:
            for (mel, attri, label), file in zip(te_ld, file_list):  # mel: 1*?*1*128*128
                mel = mel.squeeze(0).to(device)
                attri = attri.squeeze(0)
                _, feat_t = netD(mel)
                recon = netG(mel)
                melz = netG(mel, outz=True)
                reconz = netG(recon, outz=True)
                feat_t = feat_t.mean(axis=0, keepdim=True).cpu().numpy()
                sec = attri[0, 0].item()
                if cal_auc:
                    domain, status = attri[0, 1].item(), label[0, 0].item()
                    domain = 'source' if domain == 0 else 'target'

                for idx, metric in id2metric.items():
                    wn = metric.split('_')[0]
                    if score_metric is None:
                        if wn == 'D':
                            dname = metric.split('_')[1]
                            score = edfunc[dname](feat_t, sec)
                        if wn == 'G':
                            dd, st, sc = tuple(metric.split('_')[1:])
                            ori = mel if dd == 'x' else melz
                            hat = recon if dd == 'x' else reconz
                            score = scfunc[sc](specfunc(stfunc[st](hat, ori)))
                    else:
                        if metric == score_metric:
                            if wn == 'D':
                                dname = metric.split('_')[1]
                                score = edfunc[dname](feat_t, sec)
                            if wn == 'G':
                                dd, st, sc = tuple(metric.split('_')[1:])
                                ori = mel if dd == 'x' else melz
                                hat = recon if dd == 'x' else reconz
                                score = scfunc[sc](specfunc(stfunc[st](hat, ori)))
                        else:
                            score = None

                    if sec not in y_true_all[idx].keys():
                        y_true_all[idx][sec] = {'source': [], 'target': []}
                        y_score_all[idx][sec] = {'source': [], 'target': []}
                        file_sec_list[idx][sec] = []
                        score_sec_list[idx][sec] = []
                    file_sec_list[idx][sec].append(file)
                    score_sec_list[idx][sec].append(score)
                    if cal_auc:
                        y_true_all[idx][sec][domain].append(status)
                        y_score_all[idx][sec][domain].append(score)
                pbar.update(1)

    if cal_auc:
        hmean_all, result = [], [[] for _ in metric2id.keys()]
        for idx in range(len(y_true_all)):  # different scores
            y_true = dict(sorted(y_true_all[idx].items(), key=lambda t: t[0]))
            y_score = dict(sorted(y_score_all[idx].items(), key=lambda t: t[0]))
            for s in y_true.keys():
                y_true_s_auc = y_true[s]['source'] + [1 for _ in np.where(np.array(y_true[s]['target']) == 1)[0]]
                y_score_s_auc = y_score[s]['source'] + [y_score[s]['target'][idx] for idx in np.where(np.array(y_true[s]['target']) == 1)[0]]
                y_true_t_auc = y_true[s]['target'] + [1 for _ in np.where(np.array(y_true[s]['source']) == 1)[0]]
                y_score_t_auc = y_score[s]['target'] + [y_score[s]['source'][idx] for idx in np.where(np.array(y_true[s]['source']) == 1)[0]]
                y_true_pauc = y_true[s]['source'] + y_true[s]['target']
                y_score_pauc = y_score[s]['source'] + y_score[s]['target']

                AUC_s = metrics.roc_auc_score(y_true_s_auc, y_score_s_auc)
                AUC_t = metrics.roc_auc_score(y_true_t_auc, y_score_t_auc)
                pAUC = metrics.roc_auc_score(y_true_pauc, y_score_pauc, max_fpr=param['detect']['p'])
                result[idx].append([AUC_s, AUC_t, pAUC])
            hmeans = scipy.stats.hmean(np.maximum(np.array(result[idx], dtype=float), sys.float_info.epsilon), axis=0)  # AUC_s, AUC_t, pAUC
            hmean = scipy.stats.hmean(np.maximum(np.array(result[idx], dtype=float), sys.float_info.epsilon), axis=None)
            hmean_all.append([hmeans[0], hmeans[1], hmeans[2], hmean])
        hmean_all = np.array(hmean_all)
        best_hmean = np.max(hmean_all[:, 3])
        best_idx = np.where(hmean_all[:, 3] == best_hmean)[0][0]
        best_metric = id2metric[best_idx]

    # save the scores
    m_id = best_idx if cal_auc else metric2id[score_metric]
    os.makedirs(os.path.join(param['detect_dir'], data_type), exist_ok=True)
    for sec in set_sec:
        score_csv_lines = []
        for file, score in zip(file_sec_list[m_id][sec], score_sec_list[m_id][sec]):
            score_csv_lines.append([file, score])
        score_pth = '{}/{}/anomaly_score_{}_section_0{}.csv'.format(
            param['detect_dir'], data_type, param['mt'], sec)
        with open(score_pth, 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(score_csv_lines)

    if cal_auc:
        stat_csv_lines = [[param['mt'], best_metric]]
        stat_csv_lines.append(['sec', 'AUC_s', 'AUC_t', 'pAUC'])
        for i, s in enumerate(set_sec):
            stat = list(map(lambda x: round(x, 4), result[best_idx][i]))
            stat_csv_lines.append([s] + stat)
        best_hmeans = list(map(lambda x: round(x, 4), hmean_all[best_idx, :3]))
        stat_csv_lines.append(['hmean'] + best_hmeans)
        stat_csv_lines.append(['hmean_all', round(best_hmean, 4)])
        os.makedirs(os.path.join(param['result_dir'], param['mt']), exist_ok=True)
        stat_pth = '{}/{}/stat.csv'.format(param['result_dir'], param['mt'])
        with open(stat_pth, 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(stat_csv_lines)
        return best_metric, best_hmean


def main():
    print('========= Test Machine Type: {} ========='.format(param['mt']))

    param['all_sec'] = []
    if 'dev' in param['train_set']:
        param['all_sec'] += [0, 1, 2]
    if 'eval' in param['train_set']:
        param['all_sec'] += [3, 4, 5]
    train_data = train_dataset(param)
    dev_test_data = test_dataset(param, 'dev', 'test')
    test_file_list = dev_test_data.get_clip_name()
    te_ld = DataLoader(dev_test_data,
                       batch_size=1,
                       shuffle=False,
                       num_workers=0)

    device = torch.device('cuda:{}'.format(param['card_id']))
    netD = Discriminator(param)
    netG = Generator(param)
    pth_file = torch.load(param['model_pth'], map_location=torch.device('cpu'))
    netD.load_state_dict(pth_file['netD'])
    netG.load_state_dict(pth_file['netG'])
    netD.to(device)
    netG.to(device)
    train_embs = get_d_aver_emb(netD, train_data, device)

    print(f"=> Recorded best hmean: {pth_file['best_hmean']:.4}")
    print('=> Detection on dev test set')
    best_metric, best_hmean = test(netD, netG, te_ld, test_file_list, train_embs,
                                   [0, 1, 2], device, 'test', score_metric=None, cal_auc=True)
    print(f'=> Best metric: {best_metric}; Best hmean: {best_hmean:.4}')


if __name__ == '__main__':
    mt_list = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    card_num = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt', choices=mt_list, default='ToyCar')
    parser.add_argument('-c', '--card_id', type=int, choices=list(range(card_num)), default=7)
    opt = parser.parse_args()

    param['card_id'] = opt.card_id
    param['mt'] = opt.mt
    param['model_pth'] = utils.get_model_pth(param)

    for dir in [param['model_dir'], param['detect_dir'], param['result_dir']]:
        os.makedirs(dir, exist_ok=True)

    main()
