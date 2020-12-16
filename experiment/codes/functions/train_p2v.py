from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from codes.data.dataloader import CustomDataLoader as DataLoader
from codes.supports.monitor import Monitor
from codes.functions.train_base import BaseTrainer
from codes.functions.loss import W2VLoss

class Probe2VecTrainer(BaseTrainer):

    def set_lossfunc(self) -> None:
        """
        Set loss function.

        Args:
            None
        Returns:
            None
        """
        self.loss_func = W2VLoss()

    def _train(self, iterator: Iterable) -> float:
        """
        Args:
            iterator (Iterable):
        Returns:
            loss (float):
        """

        monitor = Monitor()
        self.model.train()

        for pair, label in tqdm(iterator):
            self.optimizer.zero_grad()

            pair_1 = self.model(pair[0])
            pair_2 = self.model(pair[1])
            minibatch_loss = self.loss_func(pair_1, pair_2, label)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(pair_1))

        loss = monitor.average_loss()
        return loss

    def _evaluate(self, iterator: Iterable) -> float:
        """
        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            accuracy (float):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for pair, label in tqdm(iterator):

                pair_1 = self.model(pair[0])
                pair_2 = self.model(pair[1])
                minibatch_loss = self.loss_func(pair_1, pair_2, label)

                monitor.store_loss(float(minibatch_loss), len(pair_1))

        loss = monitor.average_loss()
        return loss

    def run(self, train_loader: Iterable, valid_loader: Iterable) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
        Returns:
            None
        """

        best_loss = np.inf # Sufficietly large
        writer = SummaryWriter(self.log_dir)

        for epoch in range(1, self.epochs+1):
            print("-"*80)
            print(f"Epoch {epoch}")
            train_loss = self._train(train_loader)
            writer.add_scalar("train_loss", train_loss, epoch)
            print(f'-> Train loss: {train_loss:.4f}')

            if epoch % self.report_every == 0:
                eval_loss = self._evaluate(valid_loader)
                writer.add_scalar("eval_loss", eval_loss, epoch)
                print(f'-> Eval loss: {eval_loss:.4f}')

                if eval_loss < best_loss:
                    print(f"Validation loss improved {best_loss:.4f} -> {eval_loss:.4f}")
                    best_loss = eval_loss
                    self._save_model()
        print("-"*80)
