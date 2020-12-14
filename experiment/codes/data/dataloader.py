import torch, random
import numpy as np
from codes.data.dataset import CustomDataset as Dataset

class CustomDataLoader(object):

    def __init__(self, dataset: Dataset, batchsize: int,
                 is_eval: bool, device: str="cpu", seed: int=1):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.batchsize = batchsize

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
        self.dataset.label = self.dataset.label[idxs]

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

        X_batch, y_batch = self.dataset[idx_s:idx_e]
        X_batch = self._numpy_to_torch(X_batch)
        y_batch = self._numpy_to_torch(y_batch)
        X_batch, y_batch = X_batch.float(), y_batch.long()
        return X_batch, y_batch

if __name__ == '__main__':
    pass
