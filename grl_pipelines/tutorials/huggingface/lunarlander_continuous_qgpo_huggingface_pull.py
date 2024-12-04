import gym

from grl.algorithms.qgpo import QGPOAlgorithm
from grl.datasets import QGPOCustomizedTensorDictDataset

from grl.utils.huggingface import pull_model_from_hub


def qgpo_pipeline():

    policy_state_dict, config = pull_model_from_hub(
        repo_id="OpenDILabCommunity/LunarLanderContinuous-v2-QGPO",
    )

    qgpo = QGPOAlgorithm(
        config,
        dataset=QGPOCustomizedTensorDictDataset(
            numpy_data_path="./data.npz",
            action_augment_num=config.train.parameter.action_augment_num,
        ),
    )

    qgpo.model.load_state_dict(policy_state_dict)

    # ---------------------------------------
    # Customized train code ↓
    # ---------------------------------------
    # qgpo.train()
    # ---------------------------------------
    # Customized train code ↑
    # ---------------------------------------

    # ---------------------------------------
    # Customized deploy code ↓
    # ---------------------------------------
    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    images = [env.render(mode="rgb_array")]
    for _ in range(config.deploy.num_deploy_steps):
        observation, reward, done, _ = env.step(agent.act(observation))
        image = env.render(mode="rgb_array")
        images.append(image)
    # save images into mp4 files
    import imageio.v3 as imageio
    import numpy as np

    images = np.array(images)
    imageio.imwrite("replay.mp4", images, fps=30, quality=8)
    # ---------------------------------------
    # Customized deploy code ↑
    # ---------------------------------------


if __name__ == "__main__":

    qgpo_pipeline()
