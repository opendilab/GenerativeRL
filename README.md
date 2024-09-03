# Generative Reinforcement Learning (GRL)

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)    
[![GitHub stars](https://img.shields.io/github/stars/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/GenerativeRL)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/graphs/contributors)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

English | [简体中文(Simplified Chinese)](https://github.com/opendilab/GenerativeRL/blob/main/README.zh.md)

**GenerativeRL**, short for Generative Reinforcement Learning, is a Python library for solving reinforcement learning (RL) problems using generative models, such as diffusion models and flow models. This library aims to provide a framework for combining the power of generative models with the decision-making capabilities of reinforcement learning algorithms.

## Outline

- [Features](#features)
- [Framework Structure](#framework-structure)
- [Integrated Generative Models](#integrated-generative-models)
- [Integrated Algorithms](#integrated-algorithms)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Benchmark experiments](#benchmark-experiments)

## Features

- Support for training, evaluation and deploying diverse generative models, including diffusion models and flow models
- Integration of generative models for state representation, action representation, policy learning and dynamic model learning in RL
- Implementation of popular RL algorithms tailored for generative models, such as Q-guided policy optimization (QGPO)
- Support for various RL environments and benchmarks
- Easy-to-use API for training and evaluation

## Framework Structure

<p align="center">
  <img src="assets/framework.png" alt="Image Description 1" width="80%" height="auto" style="margin: 0 1%;">
</p>

## Integrated Generative Models

|                                                                                     | [Score Matching](https://ieeexplore.ieee.org/document/6795935) | [Flow Matching](https://arxiv.org/abs/2210.02747) |
|-------------------------------------------------------------------------------------| -------------------------------------------------------------- | ------------------------------------------------- |
| **Diffusion Model**   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18yHUAmcMh_7xq2U6TBCtcLKX2y4YvNyk)    |             |         |
| [Linear VP SDE](https://arxiv.org/abs/2011.13456)                                   | ✔                                                              | ✔                                                |
| [Generalized VP SDE](https://arxiv.org/abs/2209.15571)                              | ✔                                                              | ✔                                                |
| [Linear SDE](https://arxiv.org/abs/2206.00364)                                      | ✔                                                              | ✔                                                |
| **Flow Model**    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vrxREVXKsSbnsv9G2CnKPVvrbFZleElI)    |            |              |
| [Independent Conditional Flow Matching](https://arxiv.org/abs/2302.00482)           |  🚫                                                            | ✔                                                |
| [Optimal Transport Conditional Flow Matching](https://arxiv.org/abs/2302.00482)     |  🚫                                                            | ✔                                                |



## Integrated Algorithms

| Algo./Models                                        | Diffusion Model                                                                                                                                             |  Flow Model            |
|---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [QGPO](https://arxiv.org/abs/2304.12824)            | ✔                                                                                                                                                           |  🚫                   |
| [SRPO](https://arxiv.org/abs/2310.07297)            | ✔                                                                                                                                                           |  🚫                   |
| GMPO                                                | ✔  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A79ueOdLvTfrytjOPyfxb6zSKXi1aePv)  | ✔                     |
| GMPG                                                | ✔  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hhMvQsrV-mruvpSCpmnsOxmCb6bMPOBq)  | ✔                     |


## Installation

```bash
pip install grl
```

Or, if you want to install from source:

```bash
git clone https://github.com/opendilab/GenerativeRL.git
cd GenerativeRL
pip install -e .
```

Or you can use the docker image:
```bash
docker pull opendilab/grl:torch2.3.0-cuda12.1-cudnn8-runtime
docker run -it --rm --gpus all opendilab/grl:torch2.3.0-cuda12.1-cudnn8-runtime /bin/bash
```

## Quick Start

Here is an example of how to train a diffusion model for Q-guided policy optimization (QGPO) in the [LunarLanderContinuous-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment using GenerativeRL.

Install the required dependencies:
```bash
pip install 'gym[box2d]==0.23.1'
```
(The gym version can be from 0.23 to 0.25 for box2d environments, but it is recommended to use 0.23.1 for compatibility with D4RL.)

Download dataset from [here](https://drive.google.com/file/d/1YnT-Oeu9LPKuS_ZqNc5kol_pMlJ1DwyG/view?usp=drive_link) and save it as `data.npz` in the current directory.

GenerativeRL uses WandB for logging. It will ask you to log in to your account when you use it. You can disable it by running:
```bash
wandb offline
```

```python
import gym

from grl.algorithms.qgpo import QGPOAlgorithm
from grl.datasets import QGPOCustomizedDataset
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.lunarlander_continuous_qgpo import config

def qgpo_pipeline(config):
    qgpo = QGPOAlgorithm(config, dataset=QGPOCustomizedDataset(numpy_data_path="./data.npz", action_augment_num=config.train.parameter.action_augment_num))
    qgpo.train()

    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))

if __name__ == '__main__':
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
```

For more detailed examples and documentation, please refer to the GenerativeRL documentation.

## Documentation

The full documentation for GenerativeRL can be found at [GenerativeRL Documentation](https://opendilab.github.io/GenerativeRL/).

## Tutorials

We provide several case tutorials to help you better understand GenerativeRL. See more at [tutorials](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials).

## Benchmark experiments

We offer some baseline experiments to evaluate the performance of generative reinforcement learning algorithms. See more at [benchmark](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/benchmark).

## Contributing

We welcome contributions to GenerativeRL! If you are interested in contributing, please refer to the [Contributing Guide](CONTRIBUTING.md).

## License

GenerativeRL is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more details.
