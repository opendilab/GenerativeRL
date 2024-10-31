# GenerativeRL 教程

[英语 (English)](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/README.md) | 简体中文

## 训练生成模型

### 扩散模型

我们提供了一个简单的 colab 笔记本，演示如何使用 `GenerativeRL` 库构建扩散模型。您可以在[这里](https://colab.research.google.com/drive/18yHUAmcMh_7xq2U6TBCtcLKX2y4YvNyk#scrollTo=aqtDAvG6cQ1V)访问笔记本。

### 流模型

我们提供了一个简单的 colab 笔记本，演示如何使用 `GenerativeRL` 库构建流模型。您可以在[这里](https://colab.research.google.com/drive/1vrxREVXKsSbnsv9G2CnKPVvrbFZleElI?usp=drive_link)访问笔记本。

## 评估生成模型

### 采样生成

我们提供了一个简单的 colab 笔记本，演示如何使用 `GenerativeRL` 库从训练有素的生成模型生成样本。您可以在[这里](https://colab.research.google.com/drive/16jQhf1BDjtToxMZ4lDxB4IwGdRmr074j?usp=sharing)访问笔记本。

### 概率密度估计

我们提供了一个简单的 colab 笔记本，演示如何使用 `GenerativeRL` 库从训练有素的生成模型估计样本的概率密度。您可以在[这里](https://colab.research.google.com/drive/1zHsW13n338YqX87AIWG26KLC4uKQL1ZP?usp=sharing)访问笔记本。

## 玩具示例教程

我们提供了几个玩具示例，演示了 `GenerativeRL` 库的特性。您可以在[这里](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/)访问示例。

### 多种生成模型

- [扩散模型](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_diffusion.py)
- [能量条件扩散模型](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_energy_condition.py)
- [独立条件流匹配模型](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_icfm.py)
- [最优输运条件流匹配模型](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_otcfm.py)
- [SF2M](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_otcfm.py)

### 生成模型应用

- [世界模型](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/applications/swiss_roll_world_model.py)

### 生成模型评估

- [似然性评估](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/metrics/swiss_roll_likelihood.py)

### ODE/SDE 求解器用法

- [DPM 求解器](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/solvers/swiss_roll_dpmsolver.py)
- [SDE 求解器](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/solvers/swiss_roll_sdesolver.py)

### GenerativeRL 的特殊用法

- [自定义神经网络模块](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/special_usages/customized_modules.py)
- [类似字典结构的数据生成](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/special_usages/dict_tensor_ode.py)
