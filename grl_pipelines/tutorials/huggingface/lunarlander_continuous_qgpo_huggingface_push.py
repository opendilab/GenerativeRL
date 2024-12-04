import gym

from grl.algorithms.qgpo import QGPOAlgorithm
from grl.datasets import QGPOCustomizedTensorDictDataset
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.lunarlander_continuous_qgpo import (
    config,
    make_config,
)
from grl.utils.huggingface import push_model_to_hub


def qgpo_pipeline(config):

    qgpo = QGPOAlgorithm(
        config,
        dataset=QGPOCustomizedTensorDictDataset(
            numpy_data_path="./data.npz",
            action_augment_num=config.train.parameter.action_augment_num,
        ),
    )

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

    push_model_to_hub(
        model=qgpo.model,
        config=make_config(device="cuda"),
        env_name="Box2d",
        task_name="LunarLanderContinuous-v2",
        algo_name="QGPO",
        repo_id="OpenDILabCommunity/LunarLanderContinuous-v2-QGPO",
        score=200.0,
        video_path="replay.mp4",
        wandb_url=None,
        usage_file="grl_pipelines/tutorials/huggingface/lunarlander_continuous_qgpo_huggingface_pull.py",
        train_file="grl_pipelines/tutorials/huggingface/lunarlander_continuous_qgpo_huggingface_push.py",
        github_repo_url="https://github.com/opendilab/GenerativeRL/",
        github_doc_model_url="https://opendilab.github.io/GenerativeRL/",
        github_doc_env_url="https://www.gymlibrary.dev/environments/box2d/lunar_lander/",
        model_description=None,
        installation_guide="pip3 install gym[box2d]==0.23.1",
        platform_info=None,
        create_repo=True,
        template_path="grl_pipelines/tutorials/huggingface/modelcard_template.md",
    )


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
