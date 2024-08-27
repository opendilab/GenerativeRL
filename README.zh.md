# 生成式强化学习
    
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab)](https://twitter.com/opendilab)    
[![GitHub stars](https://img.shields.io/github/stars/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/GenerativeRL)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/GenerativeRL)](https://github.com/opendilab/GenerativeRL/graphs/contributors)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[英语 (English)](https://github.com/opendilab/GenerativeRL/blob/main/README.md) | 简体中文

**GenerativeRL** 是一个使用生成式模型解决强化学习问题的算法库，支持扩散模型和流模型等不同类型的生成式模型。这个库旨在提供一个框架，将生成式模型的能力与强化学习算法的决策能力相结合。

## 大纲

- [特性](#特性)
- [框架结构](#框架结构)
- [已集成的生成式模型](#已集成的生成式模型)
- [已集成的生成式强化学习算法](#已集成的生成式强化学习算法)
- [安装](#安装)
- [快速开始](#快速开始)
- [教程](#教程)
- [基线实验](#基线实验)

## 特性

- 支持多种扩散模型和流模型等不同类型的生成式模型的训练、评估和部署
- 在强化学习算法中集成生成式模型，用于状态与动作表示，策略学习与环境模型的学习
- 实现了多种基于生成式模型的强化学习算法
- 支持多种强化学习环境和基准
- 易于使用的训练和评估 API

## 框架结构

<p align="center">
  <img src="assets/framework.png" alt="Image Description 1" width="80%" height="auto" style="margin: 0 1%;">
</p>

## 已集成的生成式模型

|                                                                                     | [Score Matching](https://ieeexplore.ieee.org/document/6795935) | [Flow Matching](https://arxiv.org/abs/2210.02747) |
|-------------------------------------------------------------------------------------| -------------------------------------------------------------- | ------------------------------------------------- |
| **扩散模型**   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18yHUAmcMh_7xq2U6TBCtcLKX2y4YvNyk)    |               |               |
| [Linear VP SDE](https://arxiv.org/abs/2011.13456)                                   | ✔                                                              | ✔                                                |
| [Generalized VP SDE](https://arxiv.org/abs/2209.15571)                              | ✔                                                              | ✔                                                |
| [Linear SDE](https://arxiv.org/abs/2206.00364)                                      | ✔                                                              | ✔                                                |
| **流模型**    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vrxREVXKsSbnsv9G2CnKPVvrbFZleElI)    |               |                |
| [Independent Conditional Flow Matching](https://arxiv.org/abs/2302.00482)           | 🚫                                                             | ✔                                                 |
| [Optimal Transport Conditional Flow Matching](https://arxiv.org/abs/2302.00482)     | 🚫                                                             | ✔                                                 |

## 已集成的生成式强化学习算法

| 算法/模型                                           | 扩散模型                                                                                                                                                               | 流模型                  |
|---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [QGPO](https://arxiv.org/abs/2304.12824)            | ✔                                                                                                                                                                    |  🚫                   |
| [SRPO](https://arxiv.org/abs/2310.07297)            | ✔                                                                                                                                                                    |  🚫                   |
| GMPO                                                | ✔  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A79ueOdLvTfrytjOPyfxb6zSKXi1aePv)           | ✔                     |
| GMPG                                                | ✔  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hhMvQsrV-mruvpSCpmnsOxmCb6bMPOBq)           | ✔                     |

## 安装

```bash
pip install grl
```

或者，如果你想从源码安装：

```bash
git clone https://github.com/opendilab/GenerativeRL.git
cd GenerativeRL
pip install -e .
```

或者你可以使用 Docker 镜像：
```bash
docker pull opendilab/grl:torch2.3.0-cuda12.1-cudnn8-runtime
docker run -it --rm --gpus all opendilab/grl:torch2.3.0-cuda12.1-cudnn8-runtime /bin/bash
```

## 启动

这是一个在 [LunarLanderContinuous-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) 环境中训练 Q-guided policy optimization (QGPO) 的扩散模型的示例。

安装所需依赖：
```bash
pip install 'gym[box2d]==0.23.1'
```
(此处的 gym 版本可以为'0.23~0.25', 但为了同时兼容 D4RL 推荐使用 '0.23.1'。)

数据集可以从 [这里](https://drive.google.com/file/d/1YnT-Oeu9LPKuS_ZqNc5kol_pMlJ1DwyG/view?usp=drive_link) 下载，请将其置于工作路径下，并命名为 `data.npz`。

GenerativeRL 使用 WandB 记录训练日志。在使用时会要求你联网登录账号，你可以通过以下方式禁用它：
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
    qgpo = QGPOAlgorithm(config, dataset=QGPOCustomizedDataset(numpy_data_path="./data.npz",))
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

更多详细示例和文档，请参考 GenerativeRL 文档。

## 教程

我们提供了一些案例教程，用于帮助你更好地了解 GenerativeRL。详见于 [tutorials](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials)。

## 基线实验

我们提供了一些基线实验，用于评估生成式强化学习算法的性能。详见于 [benchmark](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/benchmark)。

## 开源支持

我们欢迎所有对 GenerativeRL 的贡献和支持！请参考 [开源贡献指南](CONTRIBUTING.md)。

## 开源协议

GenerativeRL 开源协议为 Apache License 2.0。更多信息和文档，请参考 [开源协议](LICENSE)。
