import os
import pickle
from typing import Iterable

import torch

import config
from execute_train import TrainExecuter
from codes.functions.evaluate_sample import SampleEvaluator as Evaluator

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True

class EvalExecuter(TrainExecuter):

    def __init__(self, eval_target: str, device: str="cpu"):

        self.data_loc = os.path.join(config.root, config.dirname,
                                     config.data_loc)

        self.param_file = os.path.join(eval_target, "params.pkl")
        self.weightfile = os.path.join(eval_target, "net.pth")

        self.args = self._load_params()
        self.evaluator = Evaluator(device)

    def _load_params(self):
        """
        Load pickled params.

        Args:
            None
        Returns:
            params
        """
        with open(self.param_file, "rb") as fp:
            params = pickle.load(fp)
        return params

    def run(self):
        """
        Run evaluation of model with test set.

        Args:
            None
        Returns:
            None
        """

        model = self._load_model()

        self.evaluator.set_model(model)
        self.evaluator.set_weight(self.weightfile)
        self.evaluator.set_optimizer(self.args.lr)
        self.evaluator.set_lossfunc()

        print("-"*80)
        print("Working on train set ...")
        train_loader = self._prepare_dataloader("train", is_eval=True)
        train_result = self.evaluator.run(train_loader)
        print(f'Loss: {train_result[0]:.4f}, accuracy: {train_result[1]:.4f}')

        print("-"*80)
        print("Working on valid set ...")
        valid_loader = self._prepare_dataloader("valid", is_eval=True)
        valid_result = self.evaluator.run(valid_loader)
        print(f'Loss: {valid_result[0]:.4f}, accuracy: {valid_result[1]:.4f}')

        print("-"*80)
        print("Working on test set ...")
        test_loader = self._prepare_dataloader("test", is_eval=True)
        test_result = self.evaluator.run(test_loader)
        print(f'Loss: {test_result[0]:.4f}, accuracy: {test_result[1]:.4f}')

if __name__ == "__main__":
    import sys

    eval_target = sys.argv[1]
    device = sys.argv[2]

    executer = EvalExecuter(eval_target, device)
    executer.run()
