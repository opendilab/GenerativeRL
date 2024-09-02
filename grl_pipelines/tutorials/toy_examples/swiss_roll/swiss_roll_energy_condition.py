################################################################################################
# This script demonstrates how to use an energy-conditioned diffusion model to train the Swiss Roll
# dataset with artificial values. We can model the energy function with a value function model and
# use the energy function to guide the diffusion process. We use a OneShotValueFunction as the value
# function model in this script.
#
# Configuration for OneShotValueFunction:
#
# value_function_model=dict(
#     device=device,
#     v_alpha=1.0,
#     DoubleVNetwork=dict(
#         state_encoder=dict(
#             type="GaussianFourierProjectionEncoder",
#             args=dict(
#                 embed_dim=128,
#                 x_shape=[x_size],
#                 scale=0.5,
#             ),
#         ),
#         backbone=dict(
#             type="ConcatenateMLP",
#             args=dict(
#                 hidden_sizes=[128 * x_size, 256, 256],
#                 output_size=1,
#                 activation="silu",
#             ),
#         ),
#     ),
# ),
#
# Then we can use the value function model to guide the diffusion process in the energy-conditioned
# diffusion model. An energy-conditioned diffusion model is a diffusion model that is conditioned on
# the energy function, which has an extra intermediate energy guidance module.
#
# Configuration for energy guidance:
#
# energy_guidance=dict(
#     t_encoder=t_encoder,
#     backbone=dict(
#         type="ConcatenateMLP",
#         args=dict(
#             hidden_sizes=[x_size + t_embedding_dim, 256, 256],
#             output_size=1,
#             activation="silu",
#         ),
#     ),
# ),
#
# We can train the energy-conditioned diffusion model with the energy guidance loss and the score
# matching loss, such as:
#
# energy_guidance_loss = energy_conditioned_diffusion_model.energy_guidance_loss(
#     x=train_fake_x,
# )
# energy_guidance_optimizer.zero_grad()
# energy_guidance_loss.backward()
# energy_guidance_optimizer.step()
#
# The fake_x is sampled from the energy-conditioned diffusion model in a way of data augmentation.
################################################################################################


import multiprocessing as mp
import os

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import Progress, track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.diffusion_model.energy_conditional_diffusion_model import (
    EnergyConditionalDiffusionModel,
)
from grl.rl_modules.value_network.one_shot_value_function import OneShotValueFunction
from grl.utils import set_seed
from grl.utils.log import log

x_size = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
data_num = 10000
config = EasyDict(
    project="energy_conditioned_diffusion_model_swiss_roll",
    dataset=dict(
        data_num=data_num,
        noise=0.6,
    ),
    model=dict(
        device=device,
        value_function_model=dict(
            device=device,
            v_alpha=1.0,
            DoubleVNetwork=dict(
                state_encoder=dict(
                    type="GaussianFourierProjectionEncoder",
                    args=dict(
                        embed_dim=128,
                        x_shape=[x_size],
                        scale=0.5,
                    ),
                ),
                backbone=dict(
                    type="ConcatenateMLP",
                    args=dict(
                        hidden_sizes=[128 * x_size, 256, 256],
                        output_size=1,
                        activation="silu",
                    ),
                ),
            ),
        ),
        diffusion_model=dict(
            device=device,
            x_size=x_size,
            alpha=1.0,
            solver=dict(
                type="ODESolver",
                args=dict(
                    library="torchdyn",
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
                            output_dim=x_size,
                            t_dim=t_embedding_dim,
                        ),
                    ),
                ),
            ),
            energy_guidance=dict(
                t_encoder=t_encoder,
                backbone=dict(
                    type="ConcatenateMLP",
                    args=dict(
                        hidden_sizes=[x_size + t_embedding_dim, 256, 256],
                        output_size=1,
                        activation="silu",
                    ),
                ),
            ),
        ),
    ),
    parameter=dict(
        unconditional_model=dict(
            batch_size=2048,
            learning_rate=5e-5,
            iterations=50000,
        ),
        support_size=data_num,
        sample_per_data=100,
        value_function_model=dict(
            batch_size=256,
            stop_training_iterations=50000,
            learning_rate=5e-4,
            discount_factor=0.99,
            update_momentum=0.995,
        ),
        energy_guidance=dict(
            batch_size=256,
            iterations=100000,
            learning_rate=5e-4,
        ),
        evaluation=dict(
            eval_freq=5000,
            video_save_path="./video-swiss-roll-energy-conditioned-diffusion-model",
            model_save_path="./model-swiss-roll-energy-conditioned-diffusion-model",
            guidance_scale=[0, 1, 2, 4, 8, 16],
        ),
    ),
)


def render_video(
    data_list, video_save_path, iteration, guidance_scale=1.0, fps=100, dpi=100
):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    fig = plt.figure(figsize=(6, 6))
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    ims = []
    colors = np.linspace(0, 1, len(data_list))

    for i, data in enumerate(data_list):
        # image alpha frm 0 to 1
        im = plt.scatter(data[:, 0], data[:, 1], s=1)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
    ani.save(
        os.path.join(
            video_save_path,
            f"iteration_{iteration}_guidance_scale_{guidance_scale}.mp4",
        ),
        fps=fps,
        dpi=dpi,
    )
    # clean up
    plt.close(fig)
    plt.clf()


def save_checkpoint(model, iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            model=model.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"checkpoint_{iteration}.pt"),
    )


def save_checkpoint(
    diffusion_model, value_model, diffusion_model_iteration, value_model_iteration, path
):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            diffusion_model=diffusion_model.state_dict(),
            value_model=value_model.state_dict(),
            diffusion_model_iteration=diffusion_model_iteration,
            value_model_iteration=value_model_iteration,
        ),
        f=os.path.join(
            path, f"checkpoint_{diffusion_model_iteration+value_model_iteration}.pt"
        ),
    )


if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")

    value_function_model = OneShotValueFunction(config.model.value_function_model).to(
        config.model.value_function_model.device
    )
    energy_conditioned_diffusion_model = EnergyConditionalDiffusionModel(
        config.model.diffusion_model, energy_model=value_function_model
    ).to(config.model.diffusion_model.device)

    value_function_model = torch.compile(value_function_model)
    energy_conditioned_diffusion_model = torch.compile(
        energy_conditioned_diffusion_model
    )

    if config.parameter.evaluation.model_save_path is not None:

        if not os.path.exists(config.parameter.evaluation.model_save_path):
            log.warning(
                f"Checkpoint path {config.parameter.evaluation.model_save_path} does not exist"
            )
            diffusion_model_iteration = 0
            value_model_iteration = 0
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.evaluation.model_save_path)
                if f.endswith(".pt")
            ]
            if len(checkpoint_files) == 0:
                log.warning(
                    f"No checkpoint files found in {config.parameter.evaluation.model_save_path}"
                )
                diffusion_model_iteration = 0
                value_model_iteration = 0
            else:
                checkpoint_files = sorted(
                    checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                checkpoint = torch.load(
                    os.path.join(
                        config.parameter.evaluation.model_save_path,
                        checkpoint_files[-1],
                    ),
                    map_location="cpu",
                )
                energy_conditioned_diffusion_model.load_state_dict(
                    checkpoint["diffusion_model"]
                )
                value_function_model.load_state_dict(checkpoint["value_model"])
                diffusion_model_iteration = checkpoint.get(
                    "diffusion_model_iteration", 0
                )
                value_model_iteration = checkpoint.get("value_model_iteration", 0)

    else:
        diffusion_model_iteration = 0
        value_model_iteration = 0

    # get data
    x_and_t = make_swiss_roll(
        n_samples=config.dataset.data_num, noise=config.dataset.noise
    )
    t = x_and_t[1].astype(np.float32)
    value = ((t - np.min(t)) / (np.max(t) - np.min(t)) - 0.5) * 5 - 4.0
    x = x_and_t[0].astype(np.float32)[:, [0, 2]]
    # transform data
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 10 - 5

    # plot data with color of value
    plt.scatter(x[:, 0], x[:, 1], c=value, vmin=-5, vmax=3)
    plt.colorbar()
    if not os.path.exists(config.parameter.evaluation.video_save_path):
        os.makedirs(config.parameter.evaluation.video_save_path)
    plt.savefig(
        os.path.join(
            config.parameter.evaluation.video_save_path, f"swiss_roll_data.png"
        )
    )
    plt.clf()

    # zip x and value
    data = np.concatenate([x, value[:, None]], axis=1)

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.unconditional_model.batch_size, shuffle=True
    )
    data_generator = get_train_data(data_loader)

    unconditional_model_optimizer = torch.optim.Adam(
        energy_conditioned_diffusion_model.model.parameters(),
        lr=config.parameter.unconditional_model.learning_rate,
    )

    moving_average_loss = 0.0

    subprocess_list = []

    for train_iter in track(
        range(config.parameter.unconditional_model.iterations),
        description="unconditional_model training",
    ):
        if train_iter < diffusion_model_iteration:
            continue

        train_data = next(data_generator).to(config.model.diffusion_model.device)
        train_x, train_value = train_data[:, :x_size], train_data[:, x_size]
        unconditional_model_training_loss = (
            energy_conditioned_diffusion_model.score_matching_loss(train_x)
        )
        unconditional_model_optimizer.zero_grad()
        unconditional_model_training_loss.backward()
        unconditional_model_optimizer.step()
        moving_average_loss = (
            0.99 * moving_average_loss + 0.01 * unconditional_model_training_loss.item()
            if train_iter > 0
            else unconditional_model_training_loss.item()
        )
        if train_iter % 100 == 0:
            log.info(
                f"iteration {train_iter}, unconditional model loss {unconditional_model_training_loss.item()}, moving average loss {moving_average_loss}"
            )

        diffusion_model_iteration = train_iter

        if (
            train_iter == 0
            or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
        ):
            energy_conditioned_diffusion_model.eval()
            for guidance_scale in [0.0]:
                t_span = torch.linspace(0.0, 1.0, 1000)
                x_t = (
                    energy_conditioned_diffusion_model.sample_forward_process(
                        t_span=t_span, batch_size=500, guidance_scale=guidance_scale
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                p = mp.Process(
                    target=render_video,
                    args=(
                        x_t,
                        config.parameter.evaluation.video_save_path,
                        f"diffusion_model_iteration_{diffusion_model_iteration}_value_model_iteration_{value_model_iteration}",
                        guidance_scale,
                        100,
                        100,
                    ),
                )
                p.start()
                subprocess_list.append(p)
            save_checkpoint(
                diffusion_model=energy_conditioned_diffusion_model,
                value_model=value_function_model,
                diffusion_model_iteration=diffusion_model_iteration,
                value_model_iteration=value_model_iteration,
                path=config.parameter.evaluation.model_save_path,
            )

    for p in subprocess_list:
        p.join()

    def generate_fake_x(model, sample_per_data):
        # model.eval()
        fake_x_sampled = []
        for i in track(
            range(config.parameter.support_size), description="Generate fake x"
        ):
            # TODO: mkae it batchsize

            fake_x_sampled.append(
                model.sample(
                    t_span=torch.linspace(0.0, 1.0, 32).to(
                        config.model.diffusion_model.device
                    ),
                    batch_size=sample_per_data,
                    guidance_scale=0.0,
                    with_grad=False,
                )
            )

        fake_x = torch.stack(fake_x_sampled, dim=0)
        return fake_x

    fake_x = generate_fake_x(
        energy_conditioned_diffusion_model, config.parameter.sample_per_data
    )

    # fake_x
    data_fake_x = fake_x.detach().cpu().numpy()

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.value_function_model.batch_size, shuffle=True
    )
    data_loader_fake_x = torch.utils.data.DataLoader(
        data_fake_x,
        batch_size=config.parameter.energy_guidance.batch_size,
        shuffle=True,
    )
    data_generator = get_train_data(data_loader)
    data_generator_fake_x = get_train_data(data_loader_fake_x)

    v_optimizer = torch.optim.Adam(
        value_function_model.v.parameters(),
        lr=config.parameter.value_function_model.learning_rate,
    )

    energy_guidance_optimizer = torch.optim.Adam(
        energy_conditioned_diffusion_model.energy_guidance.parameters(),
        lr=config.parameter.energy_guidance.learning_rate,
    )

    moving_average_v_loss = 0.0
    moving_average_energy_guidance_loss = 0.0

    subprocess_list = []

    with Progress() as progress:
        value_training = progress.add_task(
            "Value training",
            total=config.parameter.value_function_model.stop_training_iterations,
        )
        energy_guidance_training = progress.add_task(
            "Energy guidance training",
            total=config.parameter.energy_guidance.iterations,
        )

        for train_iter in range(config.parameter.energy_guidance.iterations):

            if train_iter < value_model_iteration:
                continue

            if train_iter % config.parameter.evaluation.eval_freq == 0:
                # mesh grid from -10 to 10
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                grid = np.meshgrid(x, y)
                grid = np.stack([grid[1], grid[0]], axis=0)
                grid_tensor = torch.tensor(grid, dtype=torch.float32).to(
                    config.model.diffusion_model.device
                )
                grid_tensor = torch.einsum("dij->ijd", grid_tensor)

                # plot value function by imshow
                grid_value = value_function_model(grid_tensor)
                # plt.imshow(torch.fliplr(grid_value).detach().cpu().numpy(), extent=(-10, 10, -10, 10))
                plt.imshow(
                    grid_value.detach().cpu().numpy(),
                    extent=(-10, 10, -10, 10),
                    vmin=-5,
                    vmax=3,
                )
                plt.colorbar()
                if not os.path.exists(config.parameter.evaluation.video_save_path):
                    os.makedirs(config.parameter.evaluation.video_save_path)
                plt.savefig(
                    os.path.join(
                        config.parameter.evaluation.video_save_path,
                        f"iteration_{train_iter}_value_function.png",
                    )
                )
                plt.clf()

            train_data = next(data_generator).to(config.model.diffusion_model.device)
            train_x, train_value = train_data[:, :x_size], train_data[:, x_size]
            train_fake_x = next(data_generator_fake_x).to(
                config.model.diffusion_model.device
            )
            if (
                train_iter
                < config.parameter.value_function_model.stop_training_iterations
            ):
                v_loss = value_function_model.v_loss(
                    state=train_x,
                    value=train_value.unsqueeze(-1),
                )

                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()
                moving_average_v_loss = (
                    0.99 * moving_average_v_loss + 0.01 * v_loss.item()
                    if train_iter > 0
                    else v_loss.item()
                )
                if train_iter % 100 == 0:
                    log.info(
                        f"iteration {train_iter}, value loss {v_loss.item()}, moving average loss {moving_average_v_loss}"
                    )

                # Update target
                for param, target_param in zip(
                    value_function_model.v.parameters(),
                    value_function_model.v_target.parameters(),
                ):
                    target_param.data.copy_(
                        config.parameter.value_function_model.update_momentum
                        * param.data
                        + (1 - config.parameter.value_function_model.update_momentum)
                        * target_param.data
                    )

                progress.update(value_training, advance=1)

            energy_guidance_loss = (
                energy_conditioned_diffusion_model.energy_guidance_loss(
                    x=train_fake_x,
                )
            )
            energy_guidance_optimizer.zero_grad()
            energy_guidance_loss.backward()
            energy_guidance_optimizer.step()
            moving_average_energy_guidance_loss = (
                0.99 * moving_average_energy_guidance_loss
                + 0.01 * energy_guidance_loss.item()
                if train_iter > 0
                else energy_guidance_loss.item()
            )
            if train_iter % 100 == 0:
                log.info(
                    f"iteration {train_iter}, energy guidance loss {energy_guidance_loss.item()}, moving average loss {moving_average_energy_guidance_loss}"
                )

            value_model_iteration = train_iter
            progress.update(energy_guidance_training, advance=1)

            if (
                train_iter == 0
                or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
            ):
                energy_conditioned_diffusion_model.eval()
                for guidance_scale in config.parameter.evaluation.guidance_scale:
                    t_span = torch.linspace(0.0, 1.0, 1000)
                    x_t = (
                        energy_conditioned_diffusion_model.sample_forward_process(
                            t_span=t_span, batch_size=500, guidance_scale=guidance_scale
                        )
                        .cpu()
                        .detach()
                    )
                    x_t = [
                        x.squeeze(0)
                        for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                    ]
                    p = mp.Process(
                        target=render_video,
                        args=(
                            x_t,
                            config.parameter.evaluation.video_save_path,
                            f"diffusion_model_iteration_{diffusion_model_iteration}_value_model_iteration_{value_model_iteration}",
                            guidance_scale,
                            100,
                            100,
                        ),
                    )
                    p.start()
                    subprocess_list.append(p)

                save_checkpoint(
                    diffusion_model=energy_conditioned_diffusion_model,
                    value_model=value_function_model,
                    diffusion_model_iteration=diffusion_model_iteration,
                    value_model_iteration=value_model_iteration,
                    path=config.parameter.evaluation.model_save_path,
                )

    for p in subprocess_list:
        p.join()


def sample_from_energy_conditioned_diffusion_model(
    energy_conditioned_diffusion_model, batch_size=500, guidance_scale=1.0
):
    t_span = torch.linspace(0.0, 1.0, 1000)
    x_t = (
        energy_conditioned_diffusion_model.sample(
            t_span=t_span, batch_size=batch_size, guidance_scale=guidance_scale
        )
        .cpu()
        .detach()
    )
