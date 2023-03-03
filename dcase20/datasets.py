import os
import numpy as np
from torch.utils.data import Dataset

import utils


class base_set:
    def __init__(self, param, set_type, data_type):
        '''
        set_type: 'dev' or 'eval' or 'deval'
        data_type: 'train' or 'test'
        '''
        self.pm = param
        self.st = set_type
        self.dt = data_type

        self.all_clip_spec, self.all_mid, self.all_label = None, None, None
        self.all_clip_name = {}

    def cal_spec(self, mt):
        clip_dir, set_name = {}, []
        if 'dev' in self.st:
            clip_dir['dev'] = os.path.join(self.pm['dataset_dir'], 'dev_data', mt, self.dt)
            set_name.append('dev')
        if 'eval' in self.st:
            clip_dir['eval'] = os.path.join(self.pm['dataset_dir'], 'eval_data', mt, self.dt)
            set_name.append('eval')

        self.set_clip_addr, set_mid, set_label = {}, {}, {}
        for set_type in set_name:
            self.set_clip_addr[set_type] = utils.get_clip_addr(clip_dir[set_type])
            set_mid[set_type] = utils.extract_mid(self.set_clip_addr[set_type], self.st, self.dt)
            set_label[set_type] = utils.generate_label(self.set_clip_addr[set_type], self.st, self.dt)
        if not(mt in self.all_clip_name.keys()):
            self.all_clip_name[mt] = {sn: [] for sn in set_name}

        for set_type in set_name:
            if self.all_mid is None:
                self.all_mid = set_mid[set_type]
                self.all_label = set_label[set_type]
            else:
                self.all_mid = np.concatenate((self.all_mid, set_mid[set_type]))
                self.all_label = np.concatenate((self.all_label, set_label[set_type]))
            self.all_clip_name[mt][set_type] = list(map(lambda f: os.path.basename(f),
                                                        self.set_clip_addr[set_type]))

        all_clip_spec = utils.generate_spec(clip_addr=self.set_clip_addr,
                                            spec=self.pm['feat']['spec'],
                                            fft_num=self.pm['feat']['fft_num'],
                                            mel_bin=self.pm['feat']['mel_bin'],
                                            frame_hop=self.pm['feat']['frame_hop'],
                                            top_dir=self.pm['spec_dir'],
                                            mt=mt,
                                            data_type=self.dt,
                                            setn=self.pm['train_set'])
        if self.all_clip_spec is None:
            self.all_clip_spec = all_clip_spec
        else:
            self.all_clip_spec = np.vstack((self.all_clip_spec, all_clip_spec))

        self.seg_per_clip = (all_clip_spec.shape[-1] - self.pm['feat']['frame_num'] + 1) // self.pm['feat']['graph_hop_f']

    def get_mid(self):
        return np.unique(self.all_mid[:]).tolist()

    def get_clip_num(self):
        return self.all_clip_spec.shape[0]


class seg_set(Dataset, base_set):
    # slice clip into segment
    def __init__(self, param, set_type, data_type='train'):
        super().__init__(param, set_type, data_type)
        assert data_type == 'train'
        for mt in param['mt']['train']:
            self.cal_spec(mt)

    def __len__(self):
        return self.all_label.shape[0] * self.seg_per_clip

    def __getitem__(self, idx):
        clip_id = idx // self.seg_per_clip
        spec_id = idx % self.seg_per_clip
        data = np.zeros((1, self.pm['feat']['mel_bin'],
                         self.pm['feat']['frame_num']), dtype=np.float32)
        data[0, :, :] = self.all_clip_spec[clip_id, :, spec_id: spec_id + self.pm['feat']['frame_num']]
        mid = self.all_mid[clip_id]
        label = self.all_label[clip_id]
        return data, mid, label

    def get_clip_name(self, mt, set_type):
        return self.all_clip_name[mt][set_type]

    def get_clip_data(self, idx):
        # fetch all segments in one clip
        data = np.zeros((self.seg_per_clip, 1, self.pm['feat']['mel_bin'], self.pm['feat']['frame_num']),
                        dtype=np.float32)
        for i in range(self.seg_per_clip):
            data[i, 0] = self.all_clip_spec[idx, :, i: i + self.pm['feat']['frame_num']]
        mid = self.all_mid[idx]
        label = self.all_label[idx]
        return data, mid, label


class clip_set(Dataset, base_set):
    # fetch all segments of a clip
    def __init__(self, param, mt, set_type, data_type):
        super().__init__(param, set_type, data_type)
        self.cal_spec(mt)

    def __len__(self):  # clip num
        return self.all_label.shape[0]

    def __getitem__(self, idx):
        data = np.zeros((self.seg_per_clip, 1, self.pm['feat']['mel_bin'],
                         self.pm['feat']['frame_num']), dtype=np.float32)
        for seg_id in range(self.seg_per_clip):
            data[seg_id, 0, :, :] = self.all_clip_spec[idx, :, seg_id: seg_id + self.pm['feat']['frame_num']]
        mid = self.all_mid[idx]  # int
        label = self.all_label[idx]  # int
        return data, mid, label
