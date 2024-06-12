import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.metric import compute_likelihood
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
config = EasyDict(
    dict(
        device=device,
        flow_model=dict(
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
                sigma=0.1,
            ),
            model=dict(
                type="velocity_function",
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
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=5e-3,
            data_num=10000,
            iterations=2000,
            batch_size=2048,
            clip_grad_norm=1.0,
            eval_freq=500,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint-swiss-roll-otcfm",
            video_save_path="./video-swiss-roll-otcfm",
            device=device,
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    flow_model = OptimalTransportConditionalFlowModel(config=config.flow_model).to(
        config.flow_model.device
    )
    flow_model = torch.compile(flow_model)

    # get data
    data = make_swiss_roll(n_samples=config.parameter.data_num, noise=0.01)[0].astype(
        np.float32
    )[:, [0, 2]]
    # transform data
    data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
    data[:, 1] = data[:, 1] / np.max(np.abs(data[:, 1]))
    data = (data - data.min()) / (data.max() - data.min())
    data = data * 10 - 5

    #
    optimizer = torch.optim.Adam(
        flow_model.parameters(),
        lr=config.parameter.lr,
    )

    if config.parameter.checkpoint_path is not None:

        if (
            not os.path.exists(config.parameter.checkpoint_path)
            or len(os.listdir(config.parameter.checkpoint_path)) == 0
        ):
            log.warning(
                f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
            )
            last_iteration = -1
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.checkpoint_path)
                if f.endswith(".pt")
            ]
            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            checkpoint = torch.load(
                os.path.join(config.parameter.checkpoint_path, checkpoint_files[-1]),
                map_location="cpu",
            )
            flow_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.batch_size, shuffle=True, drop_last=True
    )

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_generator = get_train_data(data_loader)

    gradient_sum = 0.0
    loss_sum = 0.0
    counter = 0
    iteration = 0

    def plot2d(data):

        plt.scatter(data[:, 0], data[:, 1])
        plt.show()

    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100):
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
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    def save_checkpoint(model, optimizer, iteration):
        if not os.path.exists(config.parameter.checkpoint_path):
            os.makedirs(config.parameter.checkpoint_path)
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                iteration=iteration,
            ),
            f=os.path.join(
                config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
            ),
        )

    history_iteration = [-1]

    def save_checkpoint_on_exit(model, optimizer, iterations):
        def exit_handler(signal, frame):
            log.info("Saving checkpoint when exit...")
            save_checkpoint(model, optimizer, iteration=iterations[-1])
            log.info("Done.")
            sys.exit(0)

        signal.signal(signal.SIGINT, exit_handler)

    save_checkpoint_on_exit(flow_model, optimizer, history_iteration)

    for iteration in track(range(config.parameter.iterations), description="Training"):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                flow_model.sample_forward_process(t_span=t_span, batch_size=500)
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(
                x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100
            )

        batch_data = next(data_generator)
        batch_data = batch_data.to(config.device)
        # plot2d(batch_data.cpu().numpy())
        flow_model.train()
        if config.parameter.training_loss_type == "flow_matching":
            x0 = flow_model.gaussian_generator(batch_data.shape[0]).to(config.device)
            loss = flow_model.flow_matching_loss(x0=x0, x1=batch_data)
        else:
            raise NotImplementedError("Unknown loss type")
        optimizer.zero_grad()
        loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            flow_model.parameters(), config.parameter.clip_grad_norm
        )
        optimizer.step()
        gradient_sum += gradien_norm.item()
        loss_sum += loss.item()
        counter += 1

        log.info(
            f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
        )

        if iteration >= 0 and iteration % 100 == 0:
            logp = compute_likelihood(
                model=flow_model,
                x=torch.tensor(data).to(config.device),
                using_Hutchinson_trace_estimator=True,
            )
            logp_mean = logp.mean()
            bits_per_dim = -logp_mean / (
                torch.prod(torch.tensor(x_size, device=config.device))
                * torch.log(torch.tensor(2.0, device=config.device))
            )
            log.info(
                f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}, log likelihood {logp_mean.item()}, bits_per_dim {bits_per_dim.item()}"
            )

            logp = compute_likelihood(
                model=flow_model,
                x=torch.tensor(data).to(config.device),
                using_Hutchinson_trace_estimator=False,
            )
            logp_mean = logp.mean()
            bits_per_dim = -logp_mean / (
                torch.prod(torch.tensor(x_size, device=config.device))
                * torch.log(torch.tensor(2.0, device=config.device))
            )
            log.info(
                f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}, log likelihood {logp_mean.item()}, bits_per_dim {bits_per_dim.item()}"
            )

        history_iteration.append(iteration)

        if iteration == config.parameter.iterations - 1:
            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                flow_model.sample_forward_process(t_span=t_span, batch_size=500)
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(
                x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100
            )

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(flow_model, optimizer, iteration)
