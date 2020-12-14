import os
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from codes.supports.monitor import Monitor
from codes.functions.evaluate_base import BaseEvaluator

class SampleEvaluator(BaseEvaluator):


    def _evaluate(self, iterator: Iterable) -> None:
        """
        Args:
            iterator (Iterable):
        Returns:
            None
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):

                y_pred = self.model(X)
                y_pred = F.softmax(y_pred, dim=-1)
                minibatch_loss = self.loss_func(y_pred, y)

                monitor.store_loss(float(minibatch_loss), len(X))
                monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        accuracy = monitor.accuracy()
        return loss, accuracy

    def run(self, loader: Iterable) -> None:
        """
        Args:
            loader (DataLoader): Dataloader for validation data.
        Returns:
            loss (float):
            accuracy (float):
        """

        loss, accuracy = self._evaluate(loader)
        return loss, accuracy
