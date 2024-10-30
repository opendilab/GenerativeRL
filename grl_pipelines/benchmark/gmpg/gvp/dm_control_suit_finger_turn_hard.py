import torch
from easydict import EasyDict
from grl.neural_network.encoders import register_encoder
import torch.nn as nn


class finger_turn_hard(nn.Module):
    def __init__(self):
        super(finger_turn_hard, self).__init__()
        self.position = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.LayerNorm(8)
        )

        self.dist_to_target = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2), nn.LayerNorm(2)
        )

        self.touch = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 4), nn.LayerNorm(4)
        )

        self.target_position = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 4), nn.LayerNorm(4)
        )

        self.velocity = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 6), nn.LayerNorm(6)
        )

    def forward(self, x: dict) -> torch.Tensor:
        if x["dist_to_target"].dim() == 1:
            dist_to_target = x["dist_to_target"].unsqueeze(-1)
        else:
            dist_to_target = x["dist_to_target"]
        position = self.position(x["position"])
        dist_to_target = self.dist_to_target(dist_to_target)
        touch = self.touch(x["touch"])
        target = self.target_position(x["target_position"])
        velocity = self.velocity(x["velocity"])
        combined_output = torch.cat(
            [position, dist_to_target, touch, target, velocity], dim=-1
        )
        return combined_output


register_encoder(finger_turn_hard, "finger_turn_hard")

data_path = ""
domain_name = "finger"
task_name = "turn_hard"
env_id = f"{domain_name}-{task_name}"
action_size = 2
state_size = 12
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
                type="finger_turn_hard",
                args=dict(),
            ),
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=action_size,
                    t_dim=t_embedding_dim,
                    condition_dim=state_size * 2,
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
                path=data_path,
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
                                hidden_sizes=[action_size + state_size * 2, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="finger_turn_hard",
                            args=dict(),
                        ),
                    ),
                    VNetwork=dict(
                        backbone=dict(
                            type="MultiLayerPerceptron",
                            args=dict(
                                hidden_sizes=[state_size * 2, 256, 256],
                                output_size=1,
                                activation="relu",
                            ),
                        ),
                        state_encoder=dict(
                            type="finger_turn_hard",
                            args=dict(),
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
                epochs=4000,
            ),
            t_span=32,
            critic=dict(
                batch_size=4096,
                epochs=8000,
                learning_rate=3e-4,
                discount_factor=0.99,
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
            checkpoint_path=f"./{project_name}/checkpoint",
            checkpoint_freq=100,
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
        env = gym.make(config.deploy.env.env_id)
        total_reward_list = []
        for i in range(100):
            observation = env.reset()
            total_reward = 0
            while True:
                # env.render()
                observation, reward, done, _ = env.step(agent.act(observation))
                total_reward += reward
                if done:
                    observation = env.reset()
                    print(f"Episode {i}, total_reward: {total_reward}")
                    total_reward_list.append(total_reward)
                    break

        print(
            f"Average total reward: {np.mean(total_reward_list)}, std: {np.std(total_reward_list)}"
        )

        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    gp_pipeline(config)
