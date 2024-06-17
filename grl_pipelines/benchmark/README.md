# Generative Reinforcement Learning Benchmark

English | [简体中文(Simplified Chinese)](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/benchmark/README.zh.md)

We evaluate the performance of policies based on generative models in reinforcement learning tasks using the [D4RL](https://arxiv.org/abs/2004.07219) dataset in an offline RL setting.

## D4RL locomotion

Previous work on using generative models to model policies in reinforcement learning tasks has yielded promising results, as follows:

| Algo.                           | [SfBC](https://arxiv.org/abs/2209.14548) |[Diffusion-QL](https://arxiv.org/abs/2208.06193)  |[QGPO](https://proceedings.mlr.press/v202/lu23d/lu23d.pdf) |[IDQL](https://arxiv.org/abs/2304.10573)|[SRPO](https://arxiv.org/abs/2310.07297)|
|-------------------------------- | ---------------------------------------- | -------------------------------------------------| --------------------------------------------------------- | -------------------------------------- | -------------------------------------- |
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

Our framework offers a more comprehensive comparison of policies based on generative models in reinforcement learning tasks, as follows:

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

Run the following command to reproduce the results:

```bash
python ./grl_pipelines/benchmark/gmpo/gvp/halfcheetah_medium_expert.py
```

## Requisites

For different RL environments, you need to install the corresponding packages. For example, to install the Mujoco and D4RL environments on an Ubuntu 20.04 system, run the following command:

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
