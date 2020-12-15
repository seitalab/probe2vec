import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, root: str, seed: int=1):
        """
        Args:
            root (str): Path to dataset directory.
        """
        gene_set = "GDS5420" # temporal

        self.root = root
        self.seed = seed

        data = self._load_data(gene_set)
        self.data = self._process_data(data)
        self.num_unique = len(np.unique(self.data.flatten()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        return data

    def _load_data(self, gene_set: str) -> np.ndarray:
        """
        Args:
            datatype (str)
        Returns:
            data (np.ndarray): Array of shape [N pairs, 2].
        """
        # targetfile = os.path.join(self.root, f"{gene_set}_co-exp.csv")
        # df_csv = pd.read_csv(targetfile)
        # data = df_csv.values
        # return data
        targetfile = os.path.join(self.root, f"{gene_set}_co-exp.pkl")
        with open(targetfile, "rb") as fp:
            data = pickle.load(fp)
        return data

    def _process_data(self, data):
        """
        Args:
            data
        Returns:
            data
        """
        # uniques = np.unique(data.flatten())
        # new_data = []
        # print(f"Processing data ...")
        # for row in tqdm(data):
        #     idx1 = np.where(uniques == row[0])[0]
        #     idx2 = np.where(uniques == row[1])[0]
        #     new_data.append([idx1, idx2])
        # new_data = np.stack(new_data)
        # print(new_data.shape)
        # return new_data
        return data


if __name__ == "__main__":
    data_loc = "/home/naokinonaka/git/probe2vec/data"
    dataset = CustomDataset(data_loc)
    print(dataset[0])
