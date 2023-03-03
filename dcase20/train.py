import os
import torch
import argparse
import yaml
import copy
import numpy as np
import time

from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn import metrics

import utils
import emb_distance as EDIS
from net import AEDC, Discriminator
from datasets import seg_set, clip_set


with open('config.yaml') as fp:
    param = yaml.safe_load(fp)


class D2GLoss(torch.nn.Module):
    '''
        Feature matching loss described in the paper.
    '''
    def __init__(self, cfg):
        super(D2GLoss, self).__init__()
        self.cfg = cfg

    def forward(self, feat_fake, feat_real):
        loss = 0
        norm_loss = {'l2': lambda x, y: F.mse_loss(x, y), 'l1': lambda x, y: F.l1_loss(x, y)}
        stat = {'mu': lambda x: x.mean(dim=0),
                'sigma': lambda x: (x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).sqrt()}

        if 'mu' in self.cfg.keys():
            mu_eff = self.cfg['mu']
            mu_fake, mu_real = stat['mu'](feat_fake), stat['mu'](feat_real)
            norm = norm_loss['l2'](mu_fake, mu_real)
            loss += mu_eff * norm
        if 'sigma' in self.cfg.keys():
            sigma_eff = self.cfg['sigma']
            sigma_fake, sigma_real = stat['sigma'](feat_fake), stat['sigma'](feat_real)
            norm = norm_loss['l2'](sigma_fake, sigma_real)
            loss += sigma_eff * norm
        return loss


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1), dtype=torch.float32, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0].view(-1, 1)
    fake = torch.ones((real_samples.shape[0], 1), dtype=torch.float32, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # gradients.shape为B*1*128*128
    gradients = gradients.view(gradients.size(0), -1)  # reshape为B*(128**2)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_one_epoch(netD, netG, tr_ld, optimD, optimG, device, d2g_eff):
    netD.train()
    netG.train()
    aver_loss, gloss_num = {'recon': 0, 'd2g': 0, 'gloss': 0}, 0
    lambda_gp = param['train']['wgan']['lambda_gp']
    MSE = torch.nn.MSELoss()
    d2g_loss = D2GLoss(param['train']['wgan']['match_item'])

    for i, (mel, _, _) in enumerate(tr_ld):
        mel = mel.to(device)
        recon = netG(mel)
        pred_real, _ = netD(mel)
        pred_fake, _ = netD(recon.detach())
        gradient_penalty = compute_gradient_penalty(netD, mel.data, recon.data, device)
        d_loss = - torch.mean(pred_real) + torch.mean(pred_fake) + lambda_gp * gradient_penalty
        optimD.zero_grad()
        d_loss.backward()
        optimD.step()

        if i % param['train']['wgan']['ncritic'] == 0:
            recon = netG(mel)
            _, feat_real = netD(mel)
            _, feat_fake = netD(recon)
            reconl = MSE(recon, mel)
            d2gl = d2g_loss(feat_fake, feat_real)
            g_loss = reconl + d2g_eff * d2gl
            optimG.zero_grad()
            g_loss.backward()
            optimG.step()

            aver_loss['recon'] += reconl.item()
            aver_loss['d2g'] += d2gl.item()
            aver_loss['gloss'] += g_loss.item()
            gloss_num = gloss_num + 1

    aver_loss['recon'] /= gloss_num
    aver_loss['d2g'] /= gloss_num
    aver_loss['gloss'] /= gloss_num
    aver_loss['gloss'] = f"{aver_loss['gloss']:.4e}"
    return netD, netG, aver_loss


def get_d_aver_emb(netD, train_set, device):
    '''
        extract embeddings of the train set via the gdconv layer of the discriminator
    '''
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


def train(netD, netG, tr_ld, te_ld, optimD, optimG, logger, device, best_aver=None):
    if best_aver is not None:
        bestD = copy.deepcopy(netD.state_dict())
        bestG = copy.deepcopy(netG.state_dict())
    d2g_eff = param['train']['wgan']['feat_match_eff']
    logger.info("============== MODEL TRAINING ==============")

    for i in range(param['train']['epoch']):
        start = time.time()

        netD, netG, aver_loss = train_one_epoch(netD, netG, tr_ld, optimD, optimG, device, d2g_eff)

        train_embs = get_d_aver_emb(netD, tr_ld.dataset, device)

        mt_aver, metric = np.zeros((len(param['mt']['test']), 3)), {}
        for j, mt in enumerate(te_ld.keys()):
            mt_aver[j], metric[mt] = test(netD, netG, te_ld[mt], train_embs, logger, device)
        aver_all_mt = np.mean(mt_aver[:, 2])
        if best_aver is None or best_aver < aver_all_mt:
            best_aver = aver_all_mt
            bestD = copy.deepcopy(netD.state_dict())
            bestG = copy.deepcopy(netG.state_dict())
        logger.info('epoch {}: [recon: {:.4e}] [d2g: {:.4e}] [gloss: {}] [best: {:.4f}] [time: {:.0f}s]'.format(
                    i, aver_loss['recon'], aver_loss['d2g'], aver_loss['gloss'], best_aver, time.time() - start))
        for j, mt in enumerate(param['mt']['test']):
            logger.info('{}: [AUC: {:.4f}] [pAUC: {:.4f}] [aver: {:.4f}] [metric: {}] '
                        .format(mt, mt_aver[j, 0], mt_aver[j, 1], mt_aver[j, 2], metric[mt]))

    torch.save({'netD': bestD, 'netG': bestG, 'best_aver': best_aver}, param['model_pth'])


# @profile
def test(netD, netG, te_ld, train_embs, logger, device):
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

    logger.info('-' * 110)
    return aver_of_all_me[best_idx, :], best_metric


def main(logger):
    train_data = seg_set(param, param['train_set'], 'train')
    param['all_mid'] = train_data.get_mid()
    tr_ld = DataLoader(train_data,
                       batch_size=param['train']['bs'],
                       shuffle=True,
                       drop_last=True,
                       num_workers=0)

    te_ld = {}
    for mt in param['mt']['test']:
        mt_test_set = clip_set(param, mt, 'dev', 'test')
        te_ld[mt] = DataLoader(mt_test_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0)

    device = torch.device('cuda:{}'.format(param['card_id']))
    netD = Discriminator(param)
    netG = AEDC(param)
    if param['resume']:
        pth_file = torch.load(utils.get_model_pth(param), map_location=torch.device('cpu'))
        netD_dict, netG_dict, best_aver = pth_file['netD'], pth_file['netG'], pth_file['best_aver']
        netD.load_state_dict(netD_dict)
        netG.load_state_dict(netG_dict)
    else:
        netD.apply(utils.weights_init)
        netG.apply(utils.weights_init)
        best_aver = None
    netD.to(device)
    netG.to(device)
    optimD = torch.optim.Adam(netD.parameters(),
                              lr=param['train']['lrD'],
                              betas=(param['train']['beta1'], 0.999))
    optimG = torch.optim.Adam(netG.parameters(),
                              lr=param['train']['lrG'],
                              betas=(param['train']['beta1'], 0.999))

    train(netD, netG, tr_ld, te_ld, optimD, optimG, logger, device, best_aver)


if __name__ == '__main__':
    mt_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    card_num = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt', choices=mt_list, default='fan')
    parser.add_argument('-c', '--card_id', type=int, choices=list(range(card_num)), default=6)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=783)
    opt = parser.parse_args()

    utils.set_seed(opt.seed)
    param['card_id'] = opt.card_id
    param['model_pth'] = utils.get_model_pth(param)
    param['resume'] = opt.resume
    param['mt'] = {'train': [opt.mt], 'test': [opt.mt]}
    for dir in [param['model_dir'], param['spec_dir'], param['log_dir']]:
        os.makedirs(dir, exist_ok=True)

    logger = utils.get_logger(param)
    logger.info(f'Seed: {opt.seed}')
    logger.info(f"Train Machine: {param['mt']['train']}")
    logger.info(f"Test Machine: {param['mt']['test']}")
    logger.info('============== TRAIN CONFIG SUMMARY ==============')
    summary = utils.config_summary(param)
    for key in summary.keys():
        message = '{}: '.format(key)
        for k, v in summary[key].items():
            message += '[{}: {}] '.format(k, summary[key][k])
        logger.info(message)

    main(logger)
