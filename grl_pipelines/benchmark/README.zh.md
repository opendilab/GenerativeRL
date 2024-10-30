# 生成式强化学习基线实验

[英语 (English)](https://github.com/zjowowen/GenerativeRL_Preview/tree/main/grl_pipelines/benchmark/README.md) | 简体中文

我们评估了使用生成式模型作为强化学习策略在 [D4RL](https://arxiv.org/abs/2004.07219) 数据集和 [RL-Unplugged DeepMind Control Suite](https://arxiv.org/abs/2006.13888) 数据集上进行离线强化学习的表现.

## D4RL locomotion

以前的相关工作的取得了不错的性能成绩，详细信息如下：

| Algo.                           | [SfBC](https://arxiv.org/abs/2209.14548) |[Diffusion-QL](https://arxiv.org/abs/2208.06193)  |[QGPO](https://proceedings.mlr.press/v202/lu23d/lu23d.pdf) |[IDQL](https://arxiv.org/abs/2304.10573)|[SRPO](https://arxiv.org/abs/2310.07297)|
|-------------------------------- | ---------- | ---------- | ---------- | --------- | --------- |
| Env./Model.                     | VPSDE                                    |  DDPM                                            | VPSDE                                                     |  DDPM                                  |  VPSDE                                 |
| halfcheetah-medium-expert-v2    | 92.6                                     |  96.8                                            | 93.5                                                      |  95.9                                  |  92.2                                  |
| hopper-medium-expert-v2         | 108.6                                    |  111.1                                           | 108.0                                                     |  108.6                                 |  100.1                                 |
| walker2d-medium-expert-v2       | 109.8                                    |  110.1                                           | 110.7                                                     |  112.7                                 |  114.0                                 |
| halfcheetah-medium-v2           | 45.9                                     |  51.1                                            | 54.1                                                      |  51.0                                  |  60.4                                  |
| hopper-medium-v2                | 57.1                                     |  90.5                                            | 98.0                                                      |  65.4                                  |  95.5                                  |
| walker2d-medium-v2              | 77.9                                     |  87.0                                            | 86.0                                                      |  82.5                                  |  84.4                                  |
| halfcheetah-medium-v2           | 37.1                                     |  47.8                                            | 47.6                                                      |  45.9                                  |  51.4                                  |
| hopper-medium-replay-v2         | 86.2                                     |  100.7                                           | 96.9                                                      |  92.1                                  |  101.2                                 |
| walker2d-medium-replay-v2       | 65.1                                     |  95.5                                            | 84.4                                                      |  85.1                                  |  84.6                                  |
| **Average**                     | 75.6                                     |  88.0                                            | 86.6                                                      |  82.1                                  |  87.1                                  |

我们的框架提供了更为全面的评估，包括不同类型的生成式模型在该任务中的表现，详细信息如下：

| Algo.                           | GMPO       | GMPO       | GMPO       | GMPG      | GMPG      | GMPG      |
|-------------------------------- | ---------- | ---------- | ---------- | --------- | --------- | --------- |
| Env./Model.                     | VPSDE      |  GVP       | ICFM       |  VPSDE    |  GVP      | ICFM      |
| halfcheetah-medium-expert-v2    | 91.8 ± 3.3 | 91.9 ± 3.2 | 83.3 ± 3.7 | 89.0 ± 6.4| 84.2 ± 8.0| 86.9 ± 4.5|
| hopper-medium-expert-v2         |111.1 ± 1.3 |112.0 ± 1.8 | 87.4 ± 25.7|107.8 ± 1.9|101.6 ± 2.9|101.7 ± 1.4|
| walker2d-medium-expert-v2       |107.7 ± 0.4 |108.1 ± 0.7 |110.3 ± 0.7 |112.8 ± 1.2|110.0 ± 1.2|110.7 ± 0.3|
| halfcheetah-medium-v2           | 49.8 ± 2.6 | 49.9 ± 2.7 | 48.0 ± 2.9 | 57.0 ± 3.1| 46.0 ± 2.7| 51.4 ± 2.9|
| hopper-medium-v2                | 71.9 ± 22.1| 74.6 ± 21.2| 69.5 ± 20.4|101.1 ± 2.6|100.1 ± 1.6| 92.8 ±18.1|
| walker2d-medium-v2              | 79.0 ± 13.2| 81.1 ± 4.3 | 79.2 ± 7.6 | 91.9 ± 0.9| 92.0 ± 1.1| 82.6 ± 2.3|
| halfcheetah-medium-v2           | 36.6 ± 2.4 | 42.3 ± 3.6 | 41.7 ± 3.2 | 50.5 ± 2.7| 39.1 ± 5.4| 41.0 ± 3.5|
| hopper-medium-replay-v2         | 89.2 ± 7.4 | 97.8 ± 3.8 | 86.0 ± 2.6 | 86.3 ±10.5|103.4 ± 2.1|104.2 ± 2.0|
| walker2d-medium-replay-v2       | 84.5 ± 4.6 | 86.4 ± 1.7 | 80.9 ± 5.3 | 90.1 ± 2.2| 81.7 ± 3.2| 79.4 ± 3.2|
| **Average**                     | 80.2 ± 4.2 | 82.7 ± 4.8 | 76.2 ± 8.0 | 87.3 ± 3.5| 84.2 ± 3.2| 83.4 ± 4.2|

运行以下命令以重现结果：

```bash
python ./grl_pipelines/benchmark/gmpo/gvp/halfcheetah_medium_expert.py
```
## D4RL AntMaze

| Algo.                           | [SfBC](https://arxiv.org/abs/2209.14548) |[Diffusion-QL](https://arxiv.org/abs/2208.06193)  |[QGPO](https://proceedings.mlr.press/v202/lu23d/lu23d.pdf) |[IDQL](https://arxiv.org/abs/2304.10573)|[SRPO](https://arxiv.org/abs/2310.07297)|
|-------------------------------- | ---------- | ---------- | ---------- | --------- | --------- |
| Env./Model.                     | VPSDE                                    |  DDPM                                            | VPSDE                                                     |  DDPM                                  |  VPSDE                                 |
| antmaze-umaze-v0                | 92.0                                     |  93.4                                            | 96.4                                                      |  94.0                                  |  97.1                                  |
| antmaze-umaze-diverse-v0        | 85.3                                     |  66.2                                            | 74.4                                                      |  80.2                                  |  82.1                                  |
| antmaze-medium-play-v0          | 81.3                                     |  76.6                                            | 83.6                                                      |  84.5                                  |  80.7                                  |
| antmaze-medium-diverse-v0       | 82.0                                     |  78.6                                            | 83.8                                                      |  84.8                                  |  75.0                                  |
| antmaze-large-play-v0           | 59.3                                     |  46.4                                            | 66.6                                                      |  63.5                                  |  53.6                                  |
| antmaze-large-diverse-v0        | 64.8                                     |  56.6                                            | 64.8                                                      |  67.9                                  |  53.6                                  |
| **Average**                     | 74.2                                     |  69.6                                            | 78.3                                                      |  79.1                                  |  73.6                                  |


| Algo.                           | GMPO       | GMPG       |
|-------------------------------- | ---------- | ---------  |
| Env./Model.                     | GVP        |  VPSDE     | 
| antmaze-umaze-v0                | 94.2 ± 0.9 | 92.5 ± 1.6 |
| antmaze-umaze-diverse-v0        | 76.8 ± 11.2| 76.0 ± 3.4 |
| antmaze-medium-play-v0          | 84.6 ± 4.2 | 62.5 ± 3.7 |
| antmaze-medium-diverse-v0       | 69.0 ± 5.6 | 67.2 ± 2.0 |
| antmaze-large-play-v0           | 49.2 ± 11.2| 40.1 ± 8.6 |
| antmaze-large-diverse-v0        | 69.4 ± 15.2| 60.5 ± 3.7 |
| **Average**                     | 73.8 ± 8.0 | 66.5 ± 3.8 |

## RL-Unplugged DeepMind Control Suite

| Algo.                           | [D4PG](https://arxiv.org/abs/1804.08617) | [RABM](https://arxiv.org/abs/2002.08396)         |[QGPO](https://proceedings.mlr.press/v202/lu23d/lu23d.pdf) |[IDQL](https://arxiv.org/abs/2304.10573)|[SRPO](https://arxiv.org/abs/2310.07297)|
|-------------------------------- | ---------- | ---------- | ---------- | --------- | --------- |
| Env./Model.                     | /                                        |  /                                               | VPSDE                                                     |  VPSDE                                 |  VPSDE                                 |
| Cartpole swingup                | 856 ± 13                                 |  798 ± 31                                        | 806 ± 54                                                  |  851 ± 9                               |  842 ± 13                              |
| Cheetah run                     | 308 ± 122                                |  304 ± 32                                        | 338 ± 135                                                 |  451 ± 231                             |  344 ± 127                             |
| Humanoid run                    | 1.72 ± 166                               |  303 ± 6                                         | 245 ±  45                                                 |  179 ±  91                             |  242 ±  22                             |
| Manipulator insert ball         | 154 ± 55                                 |  409 ± 5                                         | 340 ± 451                                                 |  308 ± 433                             |  352 ± 458                             |
| Walker stand                    | 930 ± 46                                 |  689 ± 14                                        | 672 ± 266                                                 |  850 ± 161                             |  946 ± 23                              |
| Finger turn hard                | 714 ± 80                                 |  433 ± 3                                         | 698 ± 352                                                 |  534 ± 417                             |  328 ± 464                             |
| Fish swim                       | 180 ± 55                                 |  504 ± 13                                        | 412 ± 297                                                 |  474 ± 248                             |  597 ± 356                             |
| Manipulator insert peg          | 50.4 ± 9.2                               |  209 ± 15                                        | 279 ± 229                                                 |  314 ± 376                             |  327 ± 383                             |
| Walker walk                     | 549 ± 366                                |  651 ± 8                                         | 791 ± 150                                                 |  887 ±  51                             |  963 ± 15                              |
| **Average**                     | 416 ± 83                                 |  487 ± 14                                        | 509 ± 220                                                 |  538 ± 224                             |  561 ± 207                             |

| Algo.                           | GMPO       | GMPG       |
|-------------------------------- | ---------- | ---------  |
| Env./Model.                     | GVP        |  GVP       | 
| Cartpole swingup                | 830 ± 51   | 858 ± 51   |
| Cheetah run                     | 359 ± 188  | 503 ± 212  |
| Humanoid run                    | 226 ± 72   | 209 ± 61   |
| Manipulator insert ball         | 402 ± 489  | 686 ± 341  |
| Walker stand                    | 593 ± 287  | 771 ± 292  |
| Finger turn hard                | 738 ± 204  | 657 ± 371  |
| Fish swim                       | 634 ± 192  | 515 ± 168  |
| Manipulator insert peg          | 398 ± 481  | 540 ± 343  |
| Walker walk                     | 869 ± 241  | 656 ± 233  |
| **Average**                     | 561 ± 243  | 599 ± 230  |

请从 Hugging Face 仓库下载 [RL-Unplugged DeepMind Control Suite](https://huggingface.co/datasets/OpenDILabCommunity/rl_unplugged_dm_control_suite) 数据集。

## 配置要求

对于不同的强化学习环境，需要安装对应的软件包。例如，在 Ubuntu 20.04 系统上安装 Mujoco 和 D4RL 环境，运行以下命令：

```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev -y
sudo apt-get install swig gcc g++ make locales dnsutils cmake -y
sudo apt-get install build-essential libgl1-mesa-dev libgl1-mesa-glx libglew-dev -y
sudo apt-get install libosmesa6-dev libglfw3 libglfw3-dev libsdl2-dev libsdl2-image-dev -y
sudo apt-get install libglm-dev libfreetype6-dev patchelf ffmpeg -y
mkdir -p /root/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C /root/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro210/bin:/root/.mujoco/mujoco210/bin
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
pip install lockfile
pip install "Cython<3.0"
```

## 新数据集的基准实验

GenerativeRL 支持对新数据集或自定义数据集进行基准测试。
您可以按照以下步骤使用 GMPO 和 GMPG 算法对您的数据集进行实验。

### 步骤 1: 准备数据集

首先，您需要准备数据集。数据集应该是一个 Numpy 文件，使用类似于如下的格式保存：

```python
import torch
import numpy as np

# 获取数据集
obs = torch.tensor([...])
action = torch.tensor([...])
next_obs = torch.tensor([...])
reward = torch.tensor([...])
done = torch.tensor([...])

# 将数据集转换为 Numpy 格式
obs_np = obs.numpy()
action_np = action.numpy()
next_obs_np = next_obs.numpy()
reward_np = reward.numpy()
done_np = done.numpy()

# 存储数据集为 Numpy 文件
np.savez('data.npz', obs=obs_np, action=action_np, next_obs=next_obs_np, reward=reward_np, done=done_np)
```

一个可用的 LunarLanderContinuous-v2 数据集可以从 [这里](https://drive.google.com/file/d/1YnT-Oeu9LPKuS_ZqNc5kol_pMlJ1DwyG/view?usp=drive_link) 下载。

### 步骤 2: 运行实验

运行以下命令以开展实验：


```python
import torch
from easydict import EasyDict

env_id = "LunarLanderContinuous-v2" #TODO: 指定环境 ID
action_size = 2 #TODO: 指定动作空间大小
state_size = 8 #TODO: 指定状态空间大小
algorithm_type = "GMPO" #TODO: 指定算法类型
solver_type = "ODESolver" #TODO: 指定求解器类型
model_type = "DiffusionModel" #TODO: 指定模型类型
generative_model_type = "GVP" #TODO: 指定生成式模型类型
path = dict(type="gvp") #TODO: 指定模型生成路径
model_loss_type = "flow_matching" #TODO: 指定模型损失类型
data_path = "./data.npz" #TODO: 指定数据集路径
project_name = f"{env_id}-{algorithm_type}-{generative_model_type}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
model = dict(
    device=device,
    x_size=action_size,
    solver=dict(
        type="ODESolver",
        args=dict(
            library="torchdiffeq",
        ),
    ),
    path=path,
    reverse_path=path,
    model=dict(
        type="velocity_function",
        args=dict(
            t_encoder=t_encoder,
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=action_size,
                    t_dim=t_embedding_dim,
                    condition_dim=state_size,
                    condition_hidden_dim=32,
                    t_condition_hidden_dim=128,
                ),
            ),
        ),
    ),
)

config = EasyDict(
    train=dict(
        project=project_name,
        device=device,
        wandb=dict(project=f"IQL-{env_id}-{algorithm_type}-{generative_model_type}"),
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id=env_id,
            ),
        ),
        dataset=dict(
            type="GPCustomizedTensorDictDataset",
            args=dict(
                env_id=env_id,
                numpy_data_path=data_path,
            ),
        ),
        model=dict(
            GPPolicy=dict(
                device=device,
                model_type=model_type,
                model_loss_type=model_loss_type,
                model=model,
                critic=dict(
                    device=device,
                    q_alpha=1.0,
                    DoubleQNetwork=dict(
                        backbone=dict(
                            type="ConcatenateMLP",
                            args=dict(
                                hidden_sizes=[action_size + state_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                    ),
                    VNetwork=dict(
                        backbone=dict(
                            type="MultiLayerPerceptron",
                            args=dict(
                                hidden_sizes=[state_size, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                    ),
                ),
            ),
            GuidedPolicy=dict(
                model_type=model_type,
                model=model,
            ),
        ),
        parameter=dict(
            algorithm_type=algorithm_type,
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=1e-4,
                epochs=0,
            ),
            t_span=32,
            critic=dict(
                batch_size=4096,
                epochs=2000,
                learning_rate=3e-4,
                discount_factor=0.99,
                update_momentum=0.005,
                tau=0.7,
                method="iql",
            ),
            guided_policy=dict(
                batch_size=4096,
                epochs=10000,
                learning_rate=1e-4,
                beta=1.0,
                weight_clamp=100,
            ),
            evaluation=dict(
                eval=True,
                repeat=10,
                interval=100,
            ),
            checkpoint_path=f"./{project_name}/checkpoint",
            checkpoint_freq=10,
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        t_span=32,
    ),
)

if __name__ == "__main__":

    import gym
    import d4rl
    import numpy as np

    from grl.algorithms.gmpg import GMPGAlgorithm
    from grl.utils.log import log

    def gp_pipeline(config):

        gp = GMPGAlgorithm(config)

        # ---------------------------------------
        # Customized train code ↓
        # ---------------------------------------
        gp.train()
        # ---------------------------------------
        # Customized train code ↑
        # ---------------------------------------

        # ---------------------------------------
        # Customized deploy code ↓
        # ---------------------------------------

        agent = gp.deploy()
        env = gym.make(config.deploy.env.env_id)
        total_reward_list = []
        for i in range(100):
            observation = env.reset()
            total_reward = 0
            while True:
                # env.render()
                observation, reward, done, _ = env.step(agent.act(observation))
                total_reward += reward
                if done:
                    observation = env.reset()
                    print(f"Episode {i}, total_reward: {total_reward}")
                    total_reward_list.append(total_reward)
                    break

        print(
            f"Average total reward: {np.mean(total_reward_list)}, std: {np.std(total_reward_list)}"
        )

        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    gp_pipeline(config)
```

为了进行性能测评，你需要将新数据集对应的环境注册到 Gym 中。你可以参考 [这里](https://www.gymlibrary.dev/) 来获得更多信息。
