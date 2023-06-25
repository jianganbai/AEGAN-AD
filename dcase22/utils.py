import os
import numpy as np
import librosa
import logging
import datetime
import random
import torch
from tqdm import tqdm

from maps import CLASS_ATTRI_ALL, CLASS_SEC_ATTRI, ATTRI_CODE


def get_clip_addr(clip_dir, ext='wav'):
    clip_addr = []
    for f in os.listdir(clip_dir):
        clip_ext = f.split('.')[-1]
        if clip_ext == ext:
            clip_addr.append(os.path.join(clip_dir, f))
    clip_addr = sorted(clip_addr)  # 0nor -> 0ano -> 1nor -> 1ano -> ...
    return clip_addr


def generate_spec(clip_addr, fft_num, mel_bin, frame_hop, top_dir,
                  mt, data_type, setn, rescale_ctl=True):
    all_clip_spec = None

    for set_type in clip_addr.keys():  # 'dev', 'eval'
        save_dir = os.path.join(top_dir, set_type, mt)
        os.makedirs(save_dir, exist_ok=True)
        raw_data_file = os.path.join(save_dir,
                                     f'{data_type}_raw_mel_{mel_bin}_{fft_num}_{frame_hop}_1.npy')

        if not os.path.exists(raw_data_file):
            for idx in tqdm(range(len(clip_addr[set_type])), desc=f'{mt}-{setn}-{data_type}'):
                clip, sr = librosa.load(clip_addr[set_type][idx], sr=None, mono=True)
                mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=fft_num,
                                                     hop_length=frame_hop, n_mels=mel_bin)
                mel_db = librosa.power_to_db(mel, ref=1)  # log-mel, (128, 313)

                if idx == 0:
                    set_clip_spec = np.zeros((len(clip_addr[set_type]) * mel_bin, mel.shape[1]), dtype=np.float32)
                set_clip_spec[idx * mel_bin:(idx + 1) * mel_bin, :] = mel_db
            np.save(raw_data_file, set_clip_spec)
        else:
            set_clip_spec = np.load(raw_data_file)
        if all_clip_spec is None:
            all_clip_spec = set_clip_spec
        else:
            all_clip_spec = np.vstack((all_clip_spec, set_clip_spec))

    frame_num_per_clip = all_clip_spec.shape[-1]
    save_dir = os.path.join(top_dir, setn, mt)
    os.makedirs(save_dir, exist_ok=True)
    scale_data_file = os.path.join(save_dir,
                                   f'train_scale_mel_{mel_bin}_{fft_num}_{frame_hop}_1.npy')
    if data_type == 'train' and rescale_ctl:  # scale to [-1,1]
        max_v = np.max(all_clip_spec)
        min_v = np.min(all_clip_spec)
        np.save(scale_data_file, [max_v, min_v])
    else:
        maxmin = np.load(scale_data_file)
        max_v, min_v = maxmin[0], maxmin[1]

    mean = (max_v + min_v) / 2
    scale = (max_v - min_v) / 2
    all_clip_spec = (all_clip_spec - mean) / scale

    all_clip_spec = all_clip_spec.reshape(-1, mel_bin, frame_num_per_clip)
    return all_clip_spec


def generate_label(clip_addr, set_type, data_type):
    label = np.zeros((len(clip_addr)), dtype=int)

    for idx in range(len(clip_addr)):
        # train：section_01_target_train_normal_0000_f-n_C.wav
        # test：section_01_target_test_normal_0029_f-n_C.wav
        if set_type == 'dev' and data_type == 'test':
            status_note = clip_addr[idx].split('/')[-1].split('_')[4]
            assert status_note in ['normal', 'anomaly']
            status = 0 if status_note == 'normal' else 1
        elif data_type == 'train':
            status = 0
        else:  # for eval test
            status = -1
        label[idx] = status
    return label


def extract_attri(clip_addr, mt, eval_te_flag=False):
    # train：section_01_target_train_normal_0000_f-n_C.wav
    # test：section_01_target_test_normal_0029_f-n_C.wav
    all_attri = [[] for _ in clip_addr]  # list of list, [sec, domain, att0, att1, att2, ...]
    attri_idx = CLASS_SEC_ATTRI[mt]
    for cid, clip in enumerate(clip_addr):
        file_name = os.path.basename(clip)[:os.path.basename(clip).index('.wav')]
        segs = file_name.split('_')
        sec, domain_note = int(segs[1][1]), segs[2]
        domain = 0 if domain_note == 'source' else 1
        # extraction of auxiliary scene labels, not used in training
        all_attri[cid].append(sec)
        if not eval_te_flag:
            all_attri[cid].append(domain)
            sec_attri = attri_idx[sec]
            for atn in CLASS_ATTRI_ALL[mt]:
                if atn not in sec_attri:
                    atv = 'none'
                else:
                    if mt == 'valve' and atn == 'v':
                        if 'v1pat' in segs and 'v2pat' in segs:
                            atv = 'v1pat_v2pat'
                        elif 'v1pat' in segs:
                            atv = 'v1pat'
                        elif 'v2pat' in segs:
                            atv = 'v2pat'
                        else:
                            atv = 'none'
                    else:
                        assert atn in segs
                        atv = segs[segs.index(atn) + 1]
                all_attri[cid].append(ATTRI_CODE[mt][atn][atv])  # value to code
    return np.array(all_attri)


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def get_model_pth(param):
    return os.path.join(param['model_dir'], f"{param['mt']}.pth")


def config_summary(param):
    summary = {}
    summary['feat'] = {'fft_num': param['feat']['fft_num'],
                       'mel_bin': param['feat']['mel_bin'],
                       'frame_hop': param['feat']['frame_hop'],
                       'graph_hop_f': param['feat']['graph_hop_f']}
    summary['set'] = {'dataset': param['train_set']}
    summary['net'] = {'act': param['net']['act'],
                      'normalize': param['net']['normalize'],
                      'nz': param['net']['nz'],
                      'ndf': param['net']['ndf'],
                      'ngf': param['net']['ngf']}
    summary['train'] = {'lrD': param['train']['lrD'],
                        'lrG': param['train']['lrG'],
                        'batch_size': param['train']['batch_size'],
                        'epoch': param['train']['epoch']}
    summary['wgan'] = param['train']['wgan']
    return summary


def get_logger(param):
    os.makedirs(os.path.join(param['log_dir'], param['mt']), exist_ok=True)
    log_name = './{logdir}/{mt}/train.log'.format(
        logdir=param['log_dir'],
        mt=param['mt'])

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='a' if param['resume'] else 'w')
    sh_form = logging.Formatter('%(message)s')
    fh_form = logging.Formatter('%(levelname)s - %(message)s')
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)
    sh.setFormatter(sh_form)
    fh.setFormatter(fh_form)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info('Train starts at: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    return logger
