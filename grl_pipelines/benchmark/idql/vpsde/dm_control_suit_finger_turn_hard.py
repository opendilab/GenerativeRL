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
algorithm = "IDQL"

project_name = f"{env_id}-{algorithm}"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
solver_type = "DPMSolver"
action_augment_num = 16

config = EasyDict(
    train=dict(
        project=project_name,
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
            IDQLPolicy=dict(
                device=device,
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
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
                    beta=0.1,
                    solver=dict(
                        type="DPMSolver",
                        args=dict(
                            order=2,
                            device=device,
                            steps=17,
                        ),
                    ),
                    path=dict(
                        type="linear_vp_sde",
                        beta_0=0.1,
                        beta_1=20.0,
                    ),
                    model=dict(
                        type="noise_function",
                        args=dict(
                            t_encoder=t_encoder,
                            condition_encoder=dict(
                                type="finger_turn_hard",
                                args=dict(),
                            ),
                            backbone=dict(
                                type="TemporalConcatenateMLPResNet",
                                args=dict(
                                    input_dim=state_size * 2 + action_size,
                                    output_dim=action_size,
                                    num_blocks=3,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=3e-4,
                epochs=4000,
            ),
            critic=dict(
                batch_size=4096,
                epochs=20000,
                learning_rate=3e-4,
                discount_factor=0.99,
                tau=0.7,
                update_momentum=0.005,
            ),
            evaluation=dict(
                evaluation_interval=1000,
                repeat=10,
                interval=1000,
            ),
            checkpoint_path=f"./{env_id}-{algorithm}",
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id=env_id,
            seed=0,
        ),
        num_deploy_steps=1000,
    ),
)

if __name__ == "__main__":

    import gym

    from grl.algorithms.idql import IDQLAlgorithm
    from grl.utils.log import log

    def idql_pipeline(config):

        idql = IDQLAlgorithm(config)

        # ---------------------------------------
        # Customized train code ↓
        # ---------------------------------------
        idql.train()
        # ---------------------------------------
        # Customized train code ↑
        # ---------------------------------------

        # ---------------------------------------
        # Customized deploy code ↓
        # ---------------------------------------
        agent = idql.deploy()
        env = gym.make(config.deploy.env.env_id)
        env.reset()
        for _ in range(config.deploy.num_deploy_steps):
            env.render()
            env.step(agent.act(env.observation))
        # ---------------------------------------
        # Customized deploy code ↑
        # ---------------------------------------

    log.info("config: \n{}".format(config))
    idql_pipeline(config)
