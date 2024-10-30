import torch
from easydict import EasyDict
import os
from tensordict import TensorDict

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["MUJOCO_GL"] = "egl"
Data_path = "/mnt/nfs3/zhangjinouwen/dataset/dm_control/dm_locomotion/rodent_gaps.npy"
domain_name = "rodent"
task_name = "gaps"
usePixel = True
useRichData = True
env_id = f"{domain_name}-{task_name}"
action_size = 38
state_size = 235
algorithm_type = "GMPG"
solver_type = "ODESolver"
model_type = "DiffusionModel"
generative_model_type = "GVP"
path = dict(type="gvp")
model_loss_type = "flow_matching"
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
            library="torchdiffeq_adjoint",
        ),
    ),
    path=path,
    reverse_path=path,
    model=dict(
        type="velocity_function",
        args=dict(
            t_encoder=t_encoder,
            condition_encoder=dict(
                type="TensorDictConcatenateEncoder",
                args=dict(
                    usePixel=usePixel,
                    useRichData=useRichData,
                ),
            ),
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
            type="DeepMindControlEnvSimulator",
            args=dict(
                domain_name=domain_name,
                task_name=task_name,
            ),
        ),
        dataset=dict(
            type="GPDeepMindControlTensorDictDataset",
            args=dict(
                path=Data_path,
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
                        state_encoder=dict(
                            type="TensorDictConcatenateEncoder",
                            args=dict(
                                usePixel=usePixel,
                                useRichData=useRichData,
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
                        state_encoder=dict(
                            type="TensorDictConcatenateEncoder",
                            args=dict(
                                usePixel=usePixel,
                                useRichData=useRichData,
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
                epochs=2000,
            ),
            t_span=32,
            critic=dict(
                batch_size=4096,
                epochs=2000,
                learning_rate=3e-4,
                discount_factor=0.9999,
                update_momentum=0.005,
                tau=0.7,
                method="iql",
            ),
            guided_policy=dict(
                batch_size=40960,
                epochs=500,
                learning_rate=1e-6,
                copy_from_basemodel=True,
                gradtime_step=1000,
                beta=4.0,
            ),
            evaluation=dict(
                eval=True,
                repeat=10,
                interval=5,
            ),
            checkpoint_path=f"/home/zjow/Project/generative_rl/rodent-gaps-GMPG-GVP/checkpoint",
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
    import numpy as np

    from grl.algorithms.gmpg import GMPGAlgorithm
    from grl.utils.log import log

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

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
        from dm_control import composer
        from dm_control.locomotion.examples import basic_rodent_2020

        def partial_observation_rodent(obs_dict):
            # Define the keys you want to keep
            keys_to_keep = [
                "walker/joints_pos",
                "walker/joints_vel",
                "walker/tendons_pos",
                "walker/tendons_vel",
                "walker/appendages_pos",
                "walker/world_zaxis",
                "walker/sensors_accelerometer",
                "walker/sensors_velocimeter",
                "walker/sensors_gyro",
                "walker/sensors_touch",
                "walker/egocentric_camera",
            ]
            # Filter the observation dictionary to only include the specified keys
            filtered_obs = {
                key: obs_dict[key] for key in keys_to_keep if key in obs_dict
            }
            return filtered_obs

        max_frame = 100

        width = 480
        height = 480
        video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

        env = basic_rodent_2020.rodent_run_gaps()
        total_reward_list = []
        for i in range(1):
            time_step = env.reset()
            observation = time_step.observation
            total_reward = 0
            for i in range(max_frame):
                # env.render()

                video[i] = np.hstack(
                    [
                        env.physics.render(height, width, camera_id=0),
                        env.physics.render(height, width, camera_id=1),
                    ]
                )

                observation = partial_observation_rodent(observation)

                for key in observation:
                    observation[key] = torch.tensor(
                        observation[key],
                        dtype=torch.float32,
                        device=config.train.model.GPPolicy.device,
                    )
                    if observation[key].dim() == 1 and observation[key].shape[0] == 1:
                        observation[key] = observation[key].unsqueeze(1)
                observation = TensorDict(observation)
                action = agent.act(observation)

                time_step = env.step(action)
                observation = time_step.observation
                reward = time_step.reward
                done = time_step.last()
                discount = time_step.discount

                total_reward += reward
                if done:
                    observation = env.reset()
                    print(f"Episode {i}, total_reward: {total_reward}")
                    total_reward_list.append(total_reward)
                    break

        fig, ax = plt.subplots()
        img = ax.imshow(video[0])

        # Function to update each frame
        def update(frame):
            img.set_data(video[frame])
            return (img,)

        # Create animation
        ani = FuncAnimation(fig, update, frames=max_frame, blit=True, interval=50)
        ani.save("rodent_locomotion.mp4", writer="ffmpeg", fps=30)
        plt.show()

        print(
            f"Average total reward: {np.mean(total_reward_list)}, std: {np.std(total_reward_list)}"
        )

        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    gp_pipeline(config)
