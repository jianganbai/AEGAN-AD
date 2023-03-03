import torch
import numpy as np
from sklearn.neighbors import (
    NearestNeighbors,
    LocalOutlierFactor,
)
from sklearn.metrics import DistanceMetric


class EmbeddingDetector():
    def __init__(self, train_embs):
        '''
        Args:
            train_embs ({sec: array}): all embeddings from the train set
        '''

        # Section id is only used for indexing. It is not used in the anomaly detection.
        self.mean_emb_per_ = {sec: {'source': {}, 'target': {}} for sec in train_embs.keys()}
        self.maha_dist = {sec: {'source': {}, 'target': {}} for sec in train_embs.keys()}
        self.clf = {sec: {'source': {}, 'target': {}} for sec in train_embs.keys()}
        self.lof = {sec: {'source': {}, 'target': {}} for sec in train_embs.keys()}
        self.mean_emb_per_cos = {sec: {'source': {}, 'target': {}} for sec in train_embs.keys()}

        all_embs = None
        for sec in train_embs.keys():
            if all_embs is None:
                all_embs = np.vstack([train_embs[sec]['source'], train_embs[sec]['target']])
            else:
                all_embs = np.vstack([all_embs, train_embs[sec]['source']])
                all_embs = np.vstack([all_embs, train_embs[sec]['target']])
        # maha
        self.mean_emb_per_ = np.mean(all_embs, axis=0)
        cov = np.cov(all_embs, rowvar=False)
        if np.isnan(cov).sum() > 0:
            raise ValueError("there is nan in the cov of train_embs")
        self.maha_dist = DistanceMetric.get_metric('mahalanobis', V=cov)
        # knn
        self.clf = NearestNeighbors(n_neighbors=2, metric='cosine')  # metric='mahalanobis'
        self.clf.fit(all_embs)
        # lof
        self.lof = LocalOutlierFactor(n_neighbors=4,
                                      contamination=1e-6,
                                      metric='cosine',
                                      novelty=True)
        self.lof.fit(all_embs)
        # cos
        self.mean_emb_per_cos = torch.from_numpy(self.mean_emb_per_)

    def delnan(self, mat):  # deal with nan
        if np.isnan(mat).sum() > 0:
            mat[np.isnan(mat)] = np.finfo('float32').max
        return mat

    def maha_score(self, test_embs, sec):
        score = self.maha_dist.pairwise([self.mean_emb_per_], test_embs)[0]
        score = self.delnan(score)
        return score

    def knn_score(self, test_embs, sec):
        score = self.clf.kneighbors(test_embs)[0].sum(-1)
        score = self.delnan(score)
        return score

    def lof_score(self, test_embs, sec):
        score = - self.lof.score_samples(test_embs)[0].sum(-1)
        score = self.delnan(score)
        return score

    def cos_score(self, test_embs, sec):
        test_embs = torch.from_numpy(test_embs)
        refer = self.mean_emb_per_cos.repeat(test_embs.shape[0], 1)
        score = 1 - torch.cosine_similarity(test_embs, refer).numpy()
        score = self.delnan(score)
        return score
