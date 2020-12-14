import numpy as np
from sklearn.metrics import f1_score

class Monitor(object):

    def __init__(self):
        self.num_data = 0
        self.total_loss = 0
        self.ytrue_record = np.array([])
        self.ypred_record = np.array([])

    def store_loss(self, loss: float, num_data: int) -> None:
        """
        Args:
            loss (float): Mini batch loss value.
            num_data (int): Number of data in mini batch.
        Returns:
            None
        """
        self.total_loss += loss
        self.num_data += num_data

    def store_result(self, y_trues: np.ndarray, y_preds: np.ndarray) -> None:
        """
        Args:
            y_trues (np.ndarray):
            y_preds (np.ndarray):
        Returns:
            None
        """
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        y_preds = np.argmax(y_preds, axis=1)

        self.ytrue_record = np.concatenate([self.ytrue_record, y_trues])
        self.ypred_record = np.concatenate([self.ypred_record, y_preds])
        assert(len(self.ytrue_record) == len(self.ypred_record))

    def accuracy(self) -> float:
        """
        Args:
            None
        Returns:
            accuracy (float):
        """
        num_sample = len(self.ytrue_record)
        num_correct = (self.ytrue_record == self.ypred_record).sum()
        accuracy = num_correct / num_sample
        return accuracy

    def average_loss(self) -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_loss / self.num_data

    def f1score_binary(self) -> float:
        """
        Args:
            None
        Returns:
            f1score (float):
        """
        f1score = f1_score(self.y_true_record, self.ypred_record)
        return f1score
