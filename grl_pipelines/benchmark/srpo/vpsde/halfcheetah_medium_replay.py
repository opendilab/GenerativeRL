import torch
from easydict import EasyDict
import d4rl

env_id = "halfcheetah-medium-replay-v2"
action_size = 6
state_size = 17
algorithm = "SRPO"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)

config = EasyDict(
    train=dict(
        project=f"{env_id}-{algorithm}",
        device=device,
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id=env_id,
            ),
        ),
        dataset=dict(
            type="GPD4RLTensorDictDataset",
            args=dict(
                env_id=env_id,
            ),
        ),
        model=dict(
            SRPOPolicy=dict(
                device=device,
                policy_model=dict(
                    state_dim=state_size,
                    action_dim=action_size,
                    layer=2,
                ),
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
                    ),
                ),
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
                    beta=0.2,
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
                ),
            )
        ),
        parameter=dict(
            behaviour_policy=dict(
                batch_size=4096,
                learning_rate=1e-4,
                iterations=2000,
            ),
            critic=dict(
                batch_size=4096,
                iterations=2000,
                learning_rate=3e-4,
                discount_factor=0.99,
                tau=0.7,
                update_momentum=0.005,
            ),
            policy=dict(
                batch_size=256,
                learning_rate=3e-4,
                tmax=2000000,
                iterations=2000,
            ),
            evaluation=dict(
                evaluation_interval=50,
                repeat=10,
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
