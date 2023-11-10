# Copyright (c) Facebook, Inc. and its affiliates.
#

from pathlib import Path
import pickle
from typing import Dict, Tuple, List
from models import TuckERT, TuckERTNT
import numpy as np
import torch
import os
pwd = os.getcwd()
DATA_PATH = os.path.join(pwd + '/data/')

class TemporalDataset(object):
    def __init__(self, name):
        self.root = Path(DATA_PATH) / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        try:
            inp_f = open(str(self.root / 'ts_diffs.pickle'), 'rb')
            self.time_diffs = torch.from_numpy(pickle.load(inp_f)).cuda().float()
            inp_f.close()
        except OSError:
            print("Assume all timestamps are regularly spaced")
            self.time_diffs = None

        try:
            e = open(str(self.root / 'event_list_all.pickle'), 'rb')
            self.events = pickle.load(e)
            e.close()

            f = open(str(self.root / 'ts_id'), 'rb')
            dictionary = pickle.load(f)
            f.close()
            self.timestamps = sorted(dictionary.keys())
        except OSError:
            print("Not using time intervals and events eval")
            self.events = None

        if self.events is None:
            inp_f = open(str(self.root / 'to_skip.pickle'), 'rb')
            self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
            inp_f.close()

    def has_intervals(self):
        return self.events is not None

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps


    def eval(self, model: TuckERTNT, split: str, n_queries: int = -1, missing_eval: str = 'both',at: Tuple[int] = (1, 3, 10)):
        if self.events is not None:
            return self.time_eval(model, split, n_queries, 'rhs', at)
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(lambda x: torch.mean((ranks <= x).float()).item(),at))))

        return mean_reciprocal_rank, hits_at