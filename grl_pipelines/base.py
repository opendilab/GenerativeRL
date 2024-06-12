from grl.algorithms.base import BaseAlgorithm
from grl.datasets import create_dataset
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.simulators.base import BaseEnv
from grl.utils.log import log
from grl_pipelines.configurations.base import config


def base_pipeline(config):
    """
    Overview:
        The base pipeline for training and deploying an algorithm.
    Arguments:
        - config (:obj:`EasyDict`): The configuration, which must contain the following keys:
            - train (:obj:`EasyDict`): The training configuration.
            - train.simulator (:obj:`EasyDict`): The training environment simulator configuration.
            - train.dataset (:obj:`EasyDict`): The training dataset configuration.
            - deploy (:obj:`EasyDict`): The deployment configuration.
            - deploy.env (:obj:`EasyDict`): The deployment environment configuration.
            - deploy.num_deploy_steps (:obj:`int`): The number of deployment steps.
    .. note::
        This pipeline is for demonstration purposes only.
    """

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    simulator = create_simulator(config.train.simulator)
    dataset = create_dataset(config.train.dataset)
    algo = BaseAlgorithm(simulator=simulator, dataset=dataset)
    algo.train(config=config.train)
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = algo.deploy(config=config.deploy)
    env = BaseEnv(config.deploy.env)
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    base_pipeline(config)
