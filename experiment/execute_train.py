import os
import torch
import torch.nn as nn
import random
from datetime import datetime
from typing import Iterable
from importlib import import_module

import config
from codes.data.dataset import CustomDataset as Dataset
from codes.data.dataloader import CustomDataLoader as DataLoader
from codes.functions.train_sample import SampleTrainer as Trainer

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True

class TrainExecuter(object):

    def __init__(self, args):

        self.args = args
        self.data_loc = os.path.join(config.root, config.dirname,
                                     config.data_loc)

        timestamp = self._get_timestamp()
        param_string = self._prepare_param_string()

        save_dir = os.path.join(config.save_dir, "model", param_string, timestamp)
        log_dir = os.path.join(config.save_dir, "logs", param_string, timestamp)

        self.trainer = Trainer(args.ep, save_dir=save_dir,
                               log_dir=log_dir, device=args.device)

    def _prepare_param_string(self) -> str:
        """
        Args:
            None
        Returns:
            param_string (str):
        """
        param_string = ""
        for key, value in self.args.__dict__.items():
            param_string += f"{key}-{value}_"
        param_string = param_string[:-1] # Remove last '_'
        return param_string

    def _get_timestamp(self) -> str:
        """
        Get timestamp in `yymmdd-hhmmss` format.

        Args:
            None
        Returns:
            timestamp (str): Time stamp in string.
        """
        timestamp = datetime.now()
        timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
        return timestamp

    def _load_model(self) -> nn.Module:
        """
        Load network architecture class from `codes.architectures`

        Args:
            None
        Returns:
            model (nn.Module):
        """
        modelfile = f"codes.architectures.{self.args.model}"
        ModelClass = import_module(modelfile)
        Model = ModelClass.__dict__[self.args.model]
        model = Model(params=self.args)
        print("Loaded {} ...".format(ModelClass.__name__))
        return model

    def _prepare_dataloader(self, datatype: str,
                            is_eval: bool=False) -> Iterable:
        """
        Prepare dataloader for training model.

        Args:
            datatype (str): Type of dataset ("train", "valid", "test").
            is_eval (bool): Dataloader as evaluation mode or not.
        Returns:
            loader (Iterable): Dataloader.
        """
        print("Preparing {} dataloader ...".format(datatype))

        dataset = Dataset(self.data_loc, datatype,
                          data_split_seed=self.args.seed)

        loader = DataLoader(dataset, self.args.bs, is_eval=is_eval,
                            device=self.args.device, seed=self.args.seed)
        return loader

    def run(self):
        """
        Run training of model.

        Args:
            None
        Returns:
            None
        """

        model = self._load_model()
        train_loader = self._prepare_dataloader("train")
        valid_loader = self._prepare_dataloader("valid", is_eval=True)

        self.trainer.set_model(model)
        self.trainer.set_optimizer(self.args.lr)
        self.trainer.set_lossfunc()

        self.trainer.save_params(self.args)
        self.trainer.run(train_loader, valid_loader)

if __name__ == "__main__":
    from hyperparams import args

    executer = TrainExecuter(args)
    executer.run()
