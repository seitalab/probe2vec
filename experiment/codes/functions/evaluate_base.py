import os
from typing import Iterable

import torch

from codes.functions.train_base import BaseTrainer

class BaseEvaluator(BaseTrainer):

    def __init__(self, device: str="cpu"):

        self.device = device
        self.model = None

    def set_weight(self, weight_file: str) -> None:
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert(self.model is not None)

        self.model.to("cpu")
        self.model.load_state_dict(torch.load(weight_file, map_location="cpu"))
        self.model.to(self.device)

    def _evaluate(self, iterator: Iterable) -> None:
        """
        Args:
            iterator (Iterable):
        Returns:
            None
        """

        raise NotImplementedError


    def run(self, loader: Iterable) -> None:
        """
        Args:
            loader (DataLoader): Dataloader for validation data.
        Returns:
            None
        """

        raise NotImplementedError
