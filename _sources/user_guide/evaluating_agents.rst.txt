How to evaluate RL agents performance
-------------------------------------------------

In GenerativeRL, the performance of reinforcement learning (RL) agents is evaluated using simulators or environments.

The class of agent is implemented as a class under the ``grl.agents`` module, which has a unified ``act`` method that takes the observation as input and returns the action.

User can evaluate the performance of an agent by running it in a simulator or environment and collecting the rewards.

.. code-block:: python

    import gym
    agent = algorithm.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))

