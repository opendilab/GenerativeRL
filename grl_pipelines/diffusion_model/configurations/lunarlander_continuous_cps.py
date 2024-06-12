import torch
from easydict import EasyDict

action_size = 2
state_size = 8
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

t_embedding_dim = 64  # CHANGE
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)

config = EasyDict(
    train=dict(
        project="LunarLanderContinuous-cps-srpo",
        device=device,
        simulator=dict(
            type="GymEnvSimulator",
            args=dict(
                env_id="LunarLanderContinuous-v2",
            ),
        ),
        dataset=dict(
            type="QGPOCustomizedDataset",
            args=dict(
                env_id="LunarLanderContinuous-v2",
                device=device,
                numpy_data_path="./data.npz",
            ),
        ),
        model=dict(
            CPSPolicy=dict(
                device=device,
                policy_model=dict(
                    state_dim=state_size,
                    action_dim=action_size,
                    layer=2,
                ),
                critic=dict(
                    device=device,
                    adim=action_size,
                    sdim=state_size,
                    layers=2,
                    update_momentum=0.95,
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
                ),
                diffusion_model=dict(
                    device=device,
                    x_size=action_size,
                    alpha=1.0,
                    beta=0.01,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq_adjoint",
                        ),
                    ),
                    path=dict(
                        type="linear",
                        beta_0=0.1,
                        beta_1=20.0,
                    ),
                    model=dict(
                        type="noise_function",
                        args=dict(
                            t_encoder=t_encoder,
                            backbone=dict(
                                type="ALLCONCATMLP",
                                args=dict(
                                    input_dim=state_size + action_size,
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
            training_loss_type="score_matching",
            behaviour_policy=dict(
                batch_size=2048,
                learning_rate=3e-4,
                iterations=600000,
            ),
            sample_per_state=16,
            critic=dict(
                batch_size=256,
                iterations=600000,
                learning_rate=3e-4,
                discount_factor=0.99,
                tau=0.7,
                moment=0.995,
            ),
            actor=dict(
                batch_size=256,
                iterations=1000000,
                learning_rate=3e-4,
            ),
            evaluation=dict(
                evaluation_interval=1000,
            ),
        ),
    ),
    deploy=dict(
        device=device,
        env=dict(
            env_id="LunarLanderContinuous-v2",
            seed=0,
        ),
        num_deploy_steps=1000,
    ),
)
