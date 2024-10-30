import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
from easydict import EasyDict
from grl.neural_network.encoders import register_encoder
import torch.nn as nn


class manipulator_insert(nn.Module):
    def __init__(self):
        super(manipulator_insert, self).__init__()
        self.arm_pos = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
        )
        self.arm_vel = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.LayerNorm(16)
        )
        self.touch = nn.Sequential(
            nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 10), nn.LayerNorm(10)
        )
        self.hand_pos = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.LayerNorm(8)
        )
        self.object_pos = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.LayerNorm(8)
        )
        self.object_vel = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 6), nn.LayerNorm(6)
        )
        self.target_pos = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.LayerNorm(8)
        )
        self.fish_swim = nn.Sequential(
            nn.Linear(26, 52), nn.ReLU(), nn.Linear(52, 52), nn.LayerNorm(52)
        )

    def forward(self, x: dict) -> torch.Tensor:
        shape = x["arm_pos"].shape
        arm_pos = self.arm_pos(x["arm_pos"].reshape(shape[0], 16))
        arm_vel = self.arm_vel(x["arm_vel"])
        touch = self.touch(x["touch"])
        hand_pos = self.hand_pos(x["hand_pos"])
        object_pos = self.object_pos(x["object_pos"])
        object_vel = self.object_vel(x["object_vel"])
        target_pos = self.target_pos(x["target_pos"])
        combined_output = torch.cat(
            [arm_pos, arm_vel, touch, hand_pos, object_pos, object_vel, target_pos],
            dim=-1,
        )
        return combined_output


register_encoder(manipulator_insert, "manipulator_insert")

data_path = ""
domain_name = "manipulator"
task_name = "insert_ball"
env_id = f"{domain_name}-{task_name}"
action_size = 5
state_size = 44
algorithm_type = "SRPO"
generative_model_type = "linear_vp_sde"
project_name = f"{domain_name}-{task_name}-{algorithm_type}-{generative_model_type}"

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
            SRPOPolicy=dict(
                device=device,
                policy_model=dict(
                    state_dim=state_size * 2,
                    action_dim=action_size,
                    layer=2,
                    state_encoder=dict(
                        type="manipulator_insert",
                        args=dict(),
                    ),
                ),
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
                            type="manipulator_insert",
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
                            type="manipulator_insert",
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
                                type="manipulator_insert",
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
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=3e-4,
                iterations=4000,
            ),
            critic=dict(
                batch_size=4096,
                iterations=5000,
                learning_rate=3e-4,
                discount_factor=0.99,
                tau=0.7,
                update_momentum=0.005,
            ),
            policy=dict(
                batch_size=256,
                learning_rate=3e-4,
                tmax=2000000,
                iterations=200000,
            ),
            evaluation=dict(
                evaluation_interval=500,
                repeat=10,
                interval=1000,
            ),
            checkpoint_path=f"/root/EXP/SRPO/DMC/manipulator-insert_ball/manipulator-insert_ball-SRPO-linear_vp_sde/checkpoint",
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

    from grl.algorithms.srpo import SRPOAlgorithm
    from grl.utils.log import log

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

    log.info("config: \n{}".format(config))
    srpo_pipeline(config)
