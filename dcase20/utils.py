import os
import numpy as np
import librosa
import logging
import datetime
import random
import torch
from tqdm import tqdm


def get_clip_addr(clip_dir, ext='wav'):
    clip_addr = []
    for f in os.listdir(clip_dir):
        clip_ext = f.split('.')[-1]
        if clip_ext == ext:
            clip_addr.append(os.path.join(clip_dir, f))
    clip_addr = sorted(clip_addr)  # 0nor -> 0ano -> 1nor -> 1ano -> ...
    return clip_addr


def generate_spec(clip_addr, spec, fft_num, mel_bin, frame_hop, top_dir,
                  mt, data_type, setn, rescale_ctl=True):
    all_clip_spec = None

    for set_type in clip_addr.keys():  # 'dev', 'eval'
        save_dir = os.path.join(top_dir, set_type, mt)
        os.makedirs(save_dir, exist_ok=True)
        raw_data_file = os.path.join(save_dir,
                                     f'{data_type}_raw_{spec}_{mel_bin}_{fft_num}_{frame_hop}_1.npy')

        if not os.path.exists(raw_data_file):
            for idx in tqdm(range(len(clip_addr[set_type]))):
                clip, sr = librosa.load(clip_addr[set_type][idx], sr=None, mono=True)
                if spec == 'mel':
                    mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=fft_num,
                                                         hop_length=frame_hop, n_mels=mel_bin)
                    mel_db = librosa.power_to_db(mel, ref=1)  # log-mel, (mel_bin, frame_num)
                    if idx == 0:
                        set_clip_spec = np.zeros((len(clip_addr[set_type]) * mel_bin, mel.shape[1]), dtype=np.float32)
                    set_clip_spec[idx * mel_bin:(idx + 1) * mel_bin, :] = mel_db
                elif spec == 'stft':
                    stft = librosa.stft(y=clip, n_fft=fft_num, hop_length=frame_hop)
                    stabs = np.abs(stft)
                    if idx == 0:
                        set_clip_spec = np.zeros((len(clip_addr[set_type]) * stabs.shape[0], stabs.shape[1]),
                                                 dtype=np.float32)
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
    label = np.zeros((len(clip_addr)), dtype=int)  # a label for each clip

    for idx in range(len(clip_addr)):
        # train：normal_id_01_00000791.wav
        # test：anomaly_id_01_00000181.wav(dev), id_05_00000252.wav(eval)
        if set_type == 'dev' and data_type == 'test':
            status_note = clip_addr[idx].split('/')[-1].split('_')[0]
            assert status_note in ['normal', 'anomaly']
            status = 0 if status_note == 'normal' else 1
        elif data_type == 'train':
            status = 0
        else:  # for eval test
            status = -1
        label[idx] = status
    return label


def extract_mid(clip_addr, set_type, data_type):
    # train：normal_id_01_00000791.wav
    # set：anomaly_id_01_00000181.wav(dev), id_05_00000252.wav(eval)
    all_mid = np.zeros((len(clip_addr)), dtype=int)  # [machine id]
    for i, clip_name in enumerate(clip_addr):
        file_name = os.path.basename(clip_name)[:os.path.basename(clip_name).index('.wav')]
        if set_type == 'eval' and data_type == 'test':
            mid = int(file_name.split('_')[1][1])
        else:
            mid = int(file_name.split('_')[2][1])
        all_mid[i] = mid
    return all_mid


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
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
    return os.path.join(param['model_dir'], f"{param['mt']['test'][0]}.pth")


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
                        'beta1': param['train']['beta1'],
                        'batch_size': param['train']['bs'],
                        'epoch': param['train']['epoch']}
    summary['wgan'] = param['train']['wgan']
    return summary


def get_logger(param):
    dir_n = f"{'|'.join(param['mt']['train'])}_{'|'.join(param['mt']['test'])}"
    os.makedirs(os.path.join(param['log_dir'], dir_n), exist_ok=True)

    log_name = './{logdir}/{dn}/train.log'.format(
        logdir=param['log_dir'],
        dn=dir_n)

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
