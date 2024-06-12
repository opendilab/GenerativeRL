from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

import wandb
from grl.agents.base import BaseAgent
from grl.datasets import create_dataset
from grl.rl_modules.simulators import create_simulator
from grl.utils.config import merge_two_dicts_into_newone


class BasePolicy(torch.nn.Module):
    """
    Overview:
        Base class for policies.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict = None):
        """
        Overview:
            Initialize the base policy.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()
        self.config = config

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass.
        Arguments:
            data (:obj:`torch.Tensor`): The input data.
        """
        pass


class BaseAlgorithm:
    """
    Overview:
        Base class for algorithms.
    Interfaces:
        ``__init__``, ``train``, ``deploy``
    """

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: torch.utils.data.Dataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the base algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment.
            dataset (:obj:`torch.utils.data.Dataset`): The dataset.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator
        self.dataset = dataset

        # ---------------------------------------
        # Customized model initialization code ↓
        # ---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        # ---------------------------------------
        # Customized model initialization code ↑
        # ---------------------------------------

    def train(self, config: EasyDict = None):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """

        if config is not None:
            config = merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )

        with wandb.init(
            project=(
                config.project if hasattr(config, "project") else __class__.__name__
            ),
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config.update(EasyDict(wandb_run.config))
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = (
                create_simulator(config.simulator)
                if hasattr(config, "simulator")
                else self.simulator
            )
            self.dataset = (
                create_dataset(config.dataset)
                if hasattr(config, "dataset")
                else self.dataset
            )

            # ---------------------------------------
            # Customized model initialization code ↓
            # ---------------------------------------

            self.model["BasePolicy"] = (
                BasePolicy(config.model.BasePolicy)
                if hasattr(config.model, "BasePolicy")
                else self.model.get("BasePolicy", None)
            )
            if torch.__version__ >= "2.0.0":
                self.model["BasePolicy"] = torch.compile(self.model["BasePolicy"])

            # ---------------------------------------
            # Customized model initialization code ↑
            # ---------------------------------------

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.parameter.batch_size,
                shuffle=True,
                collate_fn=None,
            )

            optimizer = torch.optim.Adam(
                self.model["BasePolicy"].parameters(),
                lr=config.parameter.learning_rate,
            )

            for train_iter in range(config.parameter.behaviour_policy.iterations):
                data = get_train_data(dataloader)
                model_training_loss = self.model["BasePolicy"](data)
                optimizer.zero_grad()
                model_training_loss.backward()
                optimizer.step()

                wandb_run.log(
                    data=dict(
                        train_iter=train_iter,
                        model_training_loss=model_training_loss.item(),
                    ),
                    commit=True,
                )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

            wandb.finish()

    def deploy(self, config: EasyDict = None) -> BaseAgent:

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)

        return BaseAgent(
            config=config,
            model=torch.nn.ModuleDict(
                {
                    "BasePolicy": self.model,
                }
            ),
        )
