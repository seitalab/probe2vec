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

class Probe2VecTrainer(BaseTrainer):

    def _train(self, iterator: Iterable) -> float:
        """
        Args:
            iterator (Iterable):
        Returns:
            loss (float):
        """

        monitor = Monitor()
        self.model.train()

        for pair in tqdm(iterator):
            self.optimizer.zero_grad()

            y_pred = self.model(X)
            y_pred = F.softmax(y_pred, dim=-1)
            minibatch_loss = self.loss_func(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(X))
            monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        accuracy = monitor.accuracy()
        return loss, accuracy

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
            for X, y in tqdm(iterator):

                y_pred = self.model(X)
                y_pred = F.softmax(y_pred, dim=-1)
                minibatch_loss = self.loss_func(y_pred, y)

                monitor.store_loss(float(minibatch_loss), len(X))
                monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        accuracy = monitor.accuracy()
        return loss, accuracy

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
            train_loss, train_acc = self._train(train_loader)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_acc, epoch)
            print(f'-> Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')

            if epoch % self.report_every == 0:
                eval_loss, eval_acc = self._evaluate(valid_loader)
                writer.add_scalar("eval_loss", eval_loss, epoch)
                writer.add_scalar("eval_accuracy", eval_acc, epoch)
                print(f'-> Eval loss: {eval_loss:.4f}, accuracy: {eval_acc:.4f}')

                if eval_loss < best_loss:
                    print(f"Validation loss improved {best_loss:.4f} -> {eval_loss:.4f}")
                    best_loss = eval_loss
                    self._save_model()
        print("-"*80)
