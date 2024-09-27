How to install GenerativeRL and its dependencies
-------------------------------------------------

GenerativeRL is a Python library that requires the following dependencies to be installed:

- Python 3.9 or higher
- PyTorch 2.0.0 or higher

Install GenerativeRL using the following command:

.. code-block:: bash

    git clone https://github.com/opendilab/GenerativeRL.git
    cd GenerativeRL
    pip install -e .

For solving reinforcement learning problems, you have to install additional environments and dependencies, such as Gym, PyBullet, MuJoCo, and DeepMind Control Suite, etc.
You can install these dependencies after installing GenerativeRL, such as:

.. code-block:: bash

    pip install gym
    pip install pybullet
    pip install mujoco-py
    pip install dm_control

It is to be noted that some of these dependencies require additional setup and licensing to use, for example, D4RL requires a special Gym environment version to be installed:

.. code-block:: bash

    pip install 'gym==0.23.1'

Another important thing is that some of the environments require additional setup, such as MuJoCo, which requires the following steps:

.. code-block:: bash

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

Check whether the installation is successful by running the following command:

.. code-block:: bash

    python -c "import grl"

