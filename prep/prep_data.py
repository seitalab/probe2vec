import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

class DataPreparator(object):

    def __init__(self, root: str, dataset_name:str) -> None:
        """
        Args:
            root (str):
            dataset_name (str):
        Returns:
            None
        """
        self.root = root
        self.dataset_name = dataset_name

        self.save_dir = os.path.join(root, "processed")
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_data(self) -> np.ndarray:
        """
        Args:
            None
        Returns:
            data (np.ndarray): Array of shape [N pairs, 2].
        """
        csvname = f"{self.dataset_name}_co-exp.csv"
        targetfile = os.path.join(self.root, csvname)
        df_csv = pd.read_csv(targetfile)
        data = df_csv.values
        return data

    def _process_data(self, data):
        """
        Args:
            data
        Returns:
            data
        """
        uniques = np.unique(data.flatten())
        new_data = []
        print(f"Processing data ...")
        for row in tqdm(data):
            idx1 = np.where(uniques == row[0])[0]
            idx2 = np.where(uniques == row[1])[0]
            new_data.append([idx1, idx2])
        new_data = np.concatenate(new_data, axis=-1)
        return new_data, uniques

    def process(self):
        """
        Args:

        Returns:

        """
        data = self._load_data()
        data, uniques = self._process_data(data)

        pickle_name = f"{self.dataset_name}_co-exp.pkl"
        savename = os.path.join(self.root, pickle_name)
        with open(savename, "wb") as fp:
            pickle.dump(data, fp)

        pickle_name_d = f"{self.dataset_name}_dict.pkl"
        savename_d = os.path.join(self.root, pickle_name_d)
        with open(savename_d, "wb") as fp:
            pickle.dump(uniques, fp)

if __name__ == "__main__":
    root = "../data"
    dataset_name = "GDS5420"
    preparator = DataPreparator(root, dataset_name)
    preparator.process()
