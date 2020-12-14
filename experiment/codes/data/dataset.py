import os
import pickle
from typing import Optional, List, Tuple

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, root: str, datatype: str,
                 data_split_seed: int=1, transform:Optional[List]=None):
        """
        Args:
            root (str): Path to dataset directory.
            datatype (str): Dataset type to load (train, valid, test)
            data_split_seed (int): Integer value for dataset split number.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "valid", "test"])

        self.root = root
        self.seed = data_split_seed

        data, label = self._load_data(datatype)
        self.data = self._process_data(data)
        self.label = self._process_label(label)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]

        # TODO: Check if transform is working.
        if self.transform:
            data = self.transform(data)

        return data, label

    def _load_data(self, datatype: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            datatype (str)
        Returns:
            X:
            y:
        """
        Xfile = f"X_{datatype}_seed{self.seed}.pkl"
        yfile = f"y_{datatype}_seed{self.seed}.pkl"
        X, y = self._open_pickle(Xfile), self._open_pickle(yfile)
        return X, y

    def _open_pickle(self, filename: str) -> np.ndarray:
        """
        Open pickled file.

        Args:
            filename (str):
        Returns:
            data (np.ndarray):
        """
        file_loc = os.path.join(self.root, filename)
        with open(file_loc, "rb") as fp:
            data = pickle.load(fp)
        return np.array(data)

    def _process_data(self, data):
        """
        Args:
            data
        Returns:
            data
        """
        data = data.astype(float)
        data /= 255.
        return data

    def _process_label(self, label):
        """
        Args:
            label
        Returns:
            label
        """
        return label

if __name__ == "__main__":
    data_loc = "/Users/naokinonaka/dev_dir/data/MNIST/processed_np"
    dataset = CustomDataset(data_loc, "valid")
    print(dataset[0])
