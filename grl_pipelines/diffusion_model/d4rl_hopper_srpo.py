import gym

from grl.algorithms.srpo import SRPOAlgorithm
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.d4rl_hopper_srpo import config


def srpo_pipeline(config):

    srpo = SRPOAlgorithm(config)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    srpo.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = srpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        env.step(agent.act(env.observation))
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    srpo_pipeline(config)
