import os
import numpy as np
from torch.utils.data import Dataset

import utils


class train_dataset(Dataset):
    def __init__(self, param):
        '''Dataset for training purpose. Output one segment at a time.

        Args:
            param (dict): hyper parameters stored in config.yaml
        '''
        clip_dir, set_name = {}, []
        if 'dev' in param['train_set']:
            clip_dir['dev'] = os.path.join(param['dataset_dir'], 'dev_data', param['mt'], 'train')
            set_name.append('dev')
        if 'eval' in param['train_set']:
            clip_dir['eval'] = os.path.join(param['dataset_dir'], 'eval_data', param['mt'], 'train')
            set_name.append('eval')
        self.param = param

        self.set_clip_addr, set_attri, set_label = {}, {}, {}
        for set_type in set_name:
            self.set_clip_addr[set_type] = utils.get_clip_addr(clip_dir[set_type])
            set_attri[set_type] = utils.extract_attri(self.set_clip_addr[set_type], param['mt'])
            set_label[set_type] = utils.generate_label(self.set_clip_addr[set_type], set_type, 'train')

        self.all_attri, self.all_label = None, None
        for set_type in set_name:
            if self.all_label is None:
                self.all_attri = set_attri[set_type]
                self.all_label = set_label[set_type]
            else:
                self.all_attri = np.vstack((self.all_attri, set_attri[set_type]))
                self.all_label = np.hstack((self.all_label, set_label[set_type]))

        print("============== TRAIN DATASET GENERATOR ==============")
        self.all_clip_spec = utils.generate_spec(clip_addr=self.set_clip_addr,
                                                 fft_num=param['feat']['fft_num'],
                                                 mel_bin=param['feat']['mel_bin'],
                                                 frame_hop=param['feat']['frame_hop'],
                                                 top_dir=param['spec_dir'],
                                                 mt=param['mt'],
                                                 data_type='train',
                                                 setn=param['train_set'])

        gn = (self.all_clip_spec.shape[-1] - param['feat']['frame_num'] + 1)
        self.graph_num_per_clip = gn // param['feat']['graph_hop_f']

    def __len__(self):
        return self.all_label.shape[0] * self.graph_num_per_clip

    def __getitem__(self, idx):  # output one segment at a time
        clip_id = idx // self.graph_num_per_clip
        spec_id = idx % self.graph_num_per_clip
        data = np.zeros((1, self.param['feat']['mel_bin'],
                         self.param['feat']['frame_num']), dtype=np.float32)
        data[0, :, :] = self.all_clip_spec[clip_id, :, spec_id: spec_id + self.param['feat']['frame_num']]
        attri = self.all_attri[clip_id]
        label = self.all_label[clip_id]
        return data, attri, label

    def get_sec(self):
        return np.unique(self.all_attri[:, 0]).tolist()

    def get_clip_name(self, set_type):
        return list(map(lambda f: os.path.basename(f), self.set_clip_addr[set_type]))

    def get_clip_num(self):
        return self.all_clip_spec.shape[0]

    def get_clip_data(self, idx):
        data = np.zeros((self.graph_num_per_clip, 1,
                         self.param['feat']['mel_bin'],
                         self.param['feat']['frame_num']), dtype=np.float32)
        for i in range(self.graph_num_per_clip):
            data[i] = self.all_clip_spec[idx, :, i: i + self.param['feat']['frame_num']]
        attri = self.all_attri[idx].reshape(1, self.all_attri.shape[1]).repeat(self.graph_num_per_clip, axis=0)
        label = self.all_label[idx].repeat(self.graph_num_per_clip, axis=0)
        return data, attri, label


class test_dataset(Dataset):
    def __init__(self, param, set_type, data_type='test'):
        '''Dataset for testing purpose. Output segments of a clip at a time.

        Args:
            param (dict): hyper parameters stored in config.yaml
            set_type (str): 'dev' or 'eval'. Two test sets are not mixed together.
            data_type (str, optional): use train data or test data for validation. Defaults to 'test'.
        '''
        clip_dir = {}
        if set_type == 'dev':
            clip_dir['dev'] = os.path.join(param['dataset_dir'], 'dev_data', param['mt'], data_type)
        if set_type == 'eval':
            clip_dir['eval'] = os.path.join(param['dataset_dir'], 'eval_data', param['mt'], data_type)
        self.param = param
        eval_te_flag = True if set_type == 'eval' and data_type == 'test' else False

        self.set_clip_addr, set_attri, set_label = {}, {}, {}
        self.set_clip_addr[set_type] = utils.get_clip_addr(clip_dir[set_type])
        set_attri[set_type] = utils.extract_attri(self.set_clip_addr[set_type], param['mt'], eval_te_flag)
        set_label[set_type] = utils.generate_label(self.set_clip_addr[set_type], set_type, data_type)

        self.set_type = set_type
        self.all_attri, self.all_label = None, None
        self.all_attri = set_attri[set_type]
        self.all_label = set_label[set_type]

        print("============== TEST DATASET GENERATOR ==============")
        self.all_clip_spec = utils.generate_spec(clip_addr=self.set_clip_addr,
                                                 fft_num=param['feat']['fft_num'],
                                                 mel_bin=param['feat']['mel_bin'],
                                                 frame_hop=param['feat']['frame_hop'],
                                                 top_dir=param['spec_dir'],
                                                 mt=param['mt'],
                                                 data_type=data_type,
                                                 setn=param['train_set'],
                                                 rescale_ctl=False)

        self.graph_num_per_clip = (self.all_clip_spec.shape[-1] - param['feat']['frame_num'] + 1) // param['feat']['graph_hop_f']

    def __len__(self):  # number of clips
        return self.all_label.shape[0]

    def __getitem__(self, idx):  # output segments of a clip at a time
        data = np.zeros((self.graph_num_per_clip, 1, self.param['feat']['mel_bin'], self.param['feat']['frame_num']), dtype=np.float32)
        for graph_id in range(self.graph_num_per_clip):
            data[graph_id, 0, :, :] = self.all_clip_spec[idx, :, graph_id: graph_id + self.param['feat']['frame_num']]
        attri = self.all_attri[idx].reshape(1, self.all_attri.shape[1]).repeat(self.graph_num_per_clip, axis=0)
        label = self.all_label[idx].repeat(self.graph_num_per_clip, axis=0)
        return data, attri, label

    def get_sec(self):
        return np.unique(self.all_attri[:, 0]).tolist()

    def get_clip_name(self):
        return list(map(lambda f: os.path.basename(f), self.set_clip_addr[self.set_type]))
