import os
import signal
import sys
import torch.multiprocessing as mp

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

from grl.generative_models.discrete_model.discrete_flow_matching import (
    DiscreteFlowMatchingModel,
)
from grl.utils import set_seed
from grl.utils.log import log
from grl.neural_network import register_module

D = 2  # dimension
S = 34  # state space


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(S, 128)
        self.net = nn.Sequential(
            nn.Linear(128 * 2 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, S * D),
        )

    def forward(self, t, x):
        # t shape: (B, 32)
        # x shape: (B, D)
        x_emb = self.embedding(x)  # (B, D, 128)
        x_emb = x_emb.reshape(x_emb.shape[0], -1)  # (B, D*128)
        x_and_t = torch.cat([x_emb, t], dim=-1)  # (B, D*128+32)
        y = self.net(x_and_t)  # (B, S*D)
        y = y.reshape(y.shape[0], D, S)  # (B, D, S)

        return y


register_module(MyModel, "MyModel")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
x_encoder = dict(
    type="DiscreteEmbeddingEncoder",
    args=dict(
        x_num=2,
        x_dim=34,
        hidden_dim=512,
    ),
)
config = EasyDict(
    dict(
        device=device,
        model=dict(
            device=device,
            variable_num=2,
            dimension=34,
            solver=dict(
                type="ODESolver",
                args=dict(
                    library="torchdiffeq",
                ),
            ),
            scheduler=dict(
                dimension=34,
                unconditional_coupling=True,
            ),
            # model=dict(
            #     type="probability_denoiser",
            #     args=dict(
            #         t_encoder=t_encoder,
            #         x_encoder=x_encoder,
            #         backbone=dict(
            #             type="TemporalSpatialResidualNet",
            #             args=dict(
            #                 hidden_sizes=[512, 256, 128],
            #                 input_dim=512,
            #                 output_dim=2*34,
            #                 t_dim=t_embedding_dim,
            #             ),
            #         ),
            #     ),
            # ),
            model=dict(
                type="probability_denoiser",
                args=dict(
                    t_encoder=t_encoder,
                    backbone=dict(
                        type="MyModel",
                        args={},
                    ),
                ),
            ),
        ),
        parameter=dict(
            lr=5e-4,
            data_num=20000,
            iterations=1000,
            batch_size=2000,
            clip_grad_norm=1.0,
            eval_freq=20,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint_discrete_flow",
            video_save_path="./video_discrete_flow",
            device=device,
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")

    # get data
    data = make_swiss_roll(n_samples=config.parameter.data_num, noise=0.4)[0].astype(
        np.float32
    )[:, [0, 2]]
    # transform data
    data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
    data[:, 1] = data[:, 1] / np.max(np.abs(data[:, 1]))
    data = (data - data.min()) / (data.max() - data.min())
    data = data

    # visialize data
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig("swiss_roll.png")
    plt.close()

    # make a meshgrid for hist2d
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    xx, yy = np.meshgrid(x, y)
    meshgrid = np.stack([xx, yy], axis=-1)

    # make a hist2d
    hist2d, _, _ = np.histogram2d(
        data[:, 1], data[:, 0], bins=32, range=[[0, 1], [0, 1]]
    )
    hist2d = hist2d / hist2d.sum()

    # visualize hist2d
    plt.figure()
    plt.pcolormesh(xx, yy, hist2d, cmap="viridis")
    # add colorbar
    plt.colorbar()
    plt.savefig("swiss_roll_hist2d.png")
    plt.close()

    # make a new dataset by transforming the original data into 2D dicrete catorical data
    data = np.floor(data * 32).astype(np.int32)

    discrete_flow_matching_model = DiscreteFlowMatchingModel(config.model).to(
        config.device
    )
    discrete_flow_matching_model = torch.compile(discrete_flow_matching_model)

    optimizer = torch.optim.Adam(
        discrete_flow_matching_model.parameters(),
        lr=config.parameter.lr,
    )

    dataloader = torch.utils.data.DataLoader(
        torch.from_numpy(data),
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )

    def render_video(data, video_save_path, iteration, fps=100, dpi=100):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))
        # plt.xlim([0, 33])
        # plt.ylim([0, 33])
        ims = []

        x = np.linspace(0, 33, 33)
        y = np.linspace(0, 33, 33)
        xx, yy = np.meshgrid(x, y)

        for i in range(len(data)):
            hist2d, _, _ = np.histogram2d(
                data[i, :, 1], data[i, :, 0], bins=32, range=[[0, 33], [0, 33]]
            )
            hist2d = hist2d / hist2d.sum()
            im = plt.pcolormesh(xx, yy, hist2d, cmap="viridis")
            # plt.colorbar()
            title = plt.text(
                0.5,
                1.05,
                f"t={i/len(data):.2f}",
                ha="center",
                va="bottom",
                transform=plt.gca().transAxes,
            )
            ims.append([im, title])

        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    p_list = []

    for i in track(range(config.parameter.iterations)):

        if i % config.parameter.eval_freq == 0:

            xt_history = discrete_flow_matching_model.sample_forward_process(
                batch_size=1000
            )
            xt_history = xt_history.cpu().numpy()
            render_video(xt_history, config.parameter.video_save_path, i)
            # p = mp.Process(target=render_video, args=(xt_history, config.parameter.video_save_path, i))
            # p.start()
            # p_list.append(p)

        loss_sum = 0
        counter = 0

        for batch in dataloader:
            optimizer.zero_grad()
            x0 = torch.ones_like(batch) * 33
            x0 = x0.to(config.parameter.device)
            batch = batch.to(config.parameter.device)
            loss = discrete_flow_matching_model.flow_matching_loss(x0=x0, x1=batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            counter += 1

        if i % config.parameter.eval_freq == 0:
            log.info(f"iteration {i}, loss {loss_sum/counter}")

        if i % config.parameter.checkpoint_freq == 0:
            if not os.path.exists(config.parameter.checkpoint_path):
                os.makedirs(config.parameter.checkpoint_path)
            torch.save(
                discrete_flow_matching_model.state_dict(),
                os.path.join(config.parameter.checkpoint_path, f"model_{i}.pth"),
            )

    for p in p_list:
        p.join()
