import random

import torch
import numpy as np

from codes.data.dataset import CustomDataset as Dataset

class Probe2VecDataLoader(object):

    def __init__(self, dataset: Dataset, batchsize: int, num_negative: int,
                 is_eval: bool, device: str="cpu", seed: int=1):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.batchsize = batchsize
        self.num_negative = num_negative

        self.shuffle = not is_eval # False if eval mode
        self.use_all = is_eval # Use all data if eval mode

        self.dataset = dataset

        # Calculate number of batch based on batchsize
        div = len(self.dataset) // self.batchsize
        mod = len(self.dataset) % self.batchsize
        if is_eval:
            # if mod == 0 => num_batch = div
            # else => num_batch = div + 1 (include last)
            self.num_batch = div + int(mod > 0)
        else:
            # if div > 0 => num_batch = div
            # else => num_batch = 1 (only use last if batch size > dataset size)
            self.num_batch = div if div > 0 else 1

        self.device = device
        self.initialize()

    def initialize(self):
        if self.shuffle:
            self._shuffle_data()
        self.itercount = 0

    def _shuffle_data(self):
        """
        Shuffle dataset
        """
        idxs = np.arange(len(self.dataset))
        random.shuffle(idxs)

        self.dataset.data = self.dataset.data[idxs]

    def _numpy_to_torch(self, data: np.ndarray):
        """
        Convert numpy array to torch tensor, and send to device.
        """
        data = torch.from_numpy(data)
        try:
            data = data.to(self.device)
        except:
            pass
        return data

    def _add_negatives(self, batch):
        """
        Args:
            batch: [bs, 2]
        Returns:
            X_rep:
            Y_rep:
            label: binary vector
        """
        X_rep = np.repeat(batch[:, 0], self.num_negative+1, axis=0)
        Y_rep, label = [], []
        for y in batch[:, 1]:
            _ys = [y]
            _label = [1]
            while len(_ys) < self.num_negative + 1:
                _y = np.random.choice(self.dataset.num_unique)
                if _y != y:
                    _ys.append(_y)
                    _label.append(0)
            Y_rep.append(_ys)
            label.append(_label)
        Y_rep = np.concatenate(Y_rep, axis=-1)
        label = np.concatenate(label, axis=-1)
        return X_rep, Y_rep, label

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        idx_s = self.itercount * self.batchsize
        idx_e = idx_s + self.batchsize

        if self.itercount == self.num_batch:
            self.initialize()
            raise StopIteration()
        else:
            self.itercount += 1

        X_batch = self.dataset[idx_s:idx_e]
        pair_1, pair_2, label = self._add_negatives(X_batch)
        pair_1 = self._numpy_to_torch(pair_1)
        pair_2 = self._numpy_to_torch(pair_2)
        label = self._numpy_to_torch(label)
        pair_1, pair_2, label = pair_1.long(), pair_2.long(), label.long()
        return pair_1, pair_2, label

if __name__ == '__main__':
    pass
