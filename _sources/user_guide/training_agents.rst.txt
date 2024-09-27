How to train and deploy reinforcement learning agents
-------------------------------------------------

In GenerativeRL, the RL algorithms are implemented as a class under the ``grl.algorithms`` module, while the agents are implemented as a class under the ``grl.agents`` module.

Every algorithm class has a ``train`` method that takes the environment, dataset, and other hyperparameters as input and returns the trained model.
Every algorithm class also has a ``deploy`` method that copys the trained model and returns the trained agent.

For training a specific RL algorithm, you need to follow these steps:

1. Create an instance of the RL algorithm class.

.. code-block:: python

    from grl.algorithms.qgpo import QGPOAlgorithm

2. Define the hyperparameters for the algorithm in a configurations dictionary. You can use the default configurations provided under the ``grl_pipelines`` module.

.. code-block:: python

    from grl_pipelines.diffusion_model.configurations.d4rl_halfcheetah_qgpo import config

3. Create an instance of algorithm class with the configurations dictionary.

.. code-block:: python

    algorithm = QGPOAlgorithm(config)

4. Train the algorithm using the ``train`` method.

.. code-block:: python

    trained_model = algorithm.train()

5. Deploy the trained model using the ``deploy`` method.

.. code-block:: python

    agent = algorithm.deploy()

6. Use the trained agent to interact with the environment and evaluate its performance.

.. code-block:: python

    import gym
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))

For more information on how to train and deploy reinforcement learning agents, please refer to the API documentation and other sections of the GenerativeRL documentation.
