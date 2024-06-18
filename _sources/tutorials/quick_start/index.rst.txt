Quick Start
===========

Generative model in GenerativeRL
---------

GenerativeRL support easy-to-use APIs for training and deploying generative model.
We provide a simple example of how to train a diffusion model on the swiss roll dataset in `Colab <https://colab.research.google.com/drive/18yHUAmcMh_7xq2U6TBCtcLKX2y4YvNyk?usp=drive_link>`_.

More usage examples can be found in the folder `grl_pipelines/tutorials/`.

Reinforcement Learning 
------------

GenerativeRL provides a simple and flexible interface for training and deploying reinforcement learning agents powered by generative models. Here's an example of how to use the library to train a Q-guided policy optimization (QGPO) agent on the HalfCheetah environment and deploy it for evaluation.

.. code-block:: python

    from grl_pipelines.diffusion_model.configurations.d4rl_halfcheetah_qgpo import config
    from grl.algorithms import QGPOAlgorithm
    from grl.utils.log import log
    import gym

    def qgpo_pipeline(config):
        qgpo = QGPOAlgorithm(config)
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

Explanation
-----------

1. First, we import the necessary components from the GenerativeRL library, including the configuration for the HalfCheetah environment and the QGPO algorithm, as well as the logging utility and the OpenAI Gym environment.

2. The ``qgpo_pipeline`` function encapsulates the training and deployment process:

   - An instance of the ``QGPOAlgorithm`` is created with the provided configuration.
   - The ``qgpo.train()`` method is called to train the QGPO agent on the HalfCheetah environment.
   - After training, the ``qgpo.deploy()`` method is called to obtain the trained agent for deployment.
   - A new instance of the HalfCheetah environment is created using ``gym.make``.
   - The environment is reset to its initial state with ``env.reset()``.
   - A loop is executed for the specified number of steps (``config.deploy.num_deploy_steps``), rendering the environment and stepping through it using the agent's ``act`` method.

3. In the ``if __name__ == '__main__'`` block, the configuration is printed to the console using the logging utility, and the ``qgpo_pipeline`` function is called with the provided configuration.

This example demonstrates how to utilize the GenerativeRL library to train a QGPO agent on the HalfCheetah environment and then deploy the trained agent for evaluation within the environment. You can modify the configuration and algorithm as needed to suit your specific use case.

For more detailed information and advanced usage examples, please refer to the API documentation and other sections of the GenerativeRL documentation.
