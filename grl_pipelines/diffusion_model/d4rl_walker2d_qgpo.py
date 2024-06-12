import gym

from grl.algorithms.qgpo import QGPOAlgorithm
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.d4rl_walker2d_qgpo import config


def qgpo_pipeline(config):

    qgpo = QGPOAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    qgpo.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
