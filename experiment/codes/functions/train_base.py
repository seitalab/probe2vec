import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BaseTrainer(object):

    def __init__(self, epochs: int, save_dir: str, log_dir: str,
                report_every: int=5, device: str="cpu"):

        self.device = device
        self.epochs = epochs
        self.report_every = report_every

        self.save_dir = save_dir
        os.makedirs(self.save_dir)
        self.log_dir = log_dir
        os.makedirs(self.log_dir)
        self.model = None

    def set_model(self, model: nn.Module) -> None:
        """
        Set model to trainer.

        Args:
            model (nn.Module)
        Returns:
            None
        """
        self.model = model.to(self.device)

    def set_optimizer(self, lr: float=1e-3) -> None:
        """
        Set optimizer for trainer.
        This function must be called after `set_model`.

        Args:
            lr (float): learning rate
        Returns:
            None
        """
        assert(self.model is not None)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def set_lossfunc(self, weight:Optional[np.ndarray]=None) -> None:
        """
        Set loss function.

        Args:
            weight (Optional[np.ndarray]):
        Returns:
            None
        """
        self.loss_func = nn.CrossEntropyLoss(weight=weight, reduction="sum")

    def save_params(self, params) -> None:
        """
        Save parameters.

        Args:
            params
        Returns:
            None
        """
        savename = self.save_dir + "/params.pkl"
        with open(savename, "wb") as fp:
            pickle.dump(params, fp)

    def _train(self, iterator):
        raise NotImplementedError

    def _evaluate(self, iterator):
        raise NotImplementedError

    def _save_model(self):
        savename = self.save_dir + "/net.pth"
        torch.save(self.model.state_dict(), savename)

    def run(self, train_loader, valid_loader):
        raise NotImplementedError
