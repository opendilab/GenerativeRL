################################################################################################
# This script demonstrates how to use edm diffusion to train Swiss Roll dataset.
################################################################################################

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
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.edm_diffusion_model.edm_diffusion_model import EDMModel
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
        edm_model=dict(  
            path=dict(
                edm_type="iDDPM_edm", # *["VP_edm", "VE_edm", "iDDPM_edm", "EDM"]
                params=dict(
                    #^ 1: VP_edm
                    # beta_d=19.9, 
                    # beta_min=0.1, 
                    # M=1000, 
                    # epsilon_t=1e-5,
                    # epsilon_s=1e-4,
                    #^ 2: VE_edm
                    # sigma_min=0.02,
                    # sigma_max=100,
                    #^ 3: iDDPM_edm
                    # C_1=0.001,
                    # C_2=0.008,
                    # M=1000,
                    #^ 4: EDM
                    # sigma_min=0.002,
                    # sigma_max=80,
                    # sigma_data=0.5,
                    # P_mean=-1.21,
                    # P_std=1.21,
                )
            ),

            solver=dict(
                solver_type="heun", 
                # *['euler', 'heun']
                params=dict(
                    num_steps=18,
                    alpha=1, 
                    S_churn=0., 
                    S_min=0., 
                    S_max=float("inf"),
                    S_noise=1.,
                    rho=7, #* EDM needs rho 
                    epsilon_s=1e-3 #* VP needs epsilon_s
                )
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
        ),
        parameter=dict(
            training_loss_type="score_matching",
            lr=5e-3,
            data_num=10000,
            iterations=1000,
            batch_size=2048,
            clip_grad_norm=1.0,
            eval_freq=500,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint",
            video_save_path="./video",
            device=device,
        ),
    )
)
if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    edm_diffusion_model = EDMModel(config=config).to(config.device)
    edm_diffusion_model = torch.compile(edm_diffusion_model)
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
        edm_diffusion_model.parameters(),
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
            edm_diffusion_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.batch_size, shuffle=True
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
        ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
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
    batch_data = next(data_generator)
    batch_data = batch_data.to(config.device)
    
    for i in range(10):
        edm_diffusion_model.train()
        loss = edm_diffusion_model(batch_data)
        optimizer.zero_grad()
        loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            edm_diffusion_model.parameters(), config.parameter.clip_grad_norm
        )
        optimizer.step()
        gradient_sum += gradien_norm.item()
        loss_sum += loss.item()
        counter += 1
        iteration += 1
        log.info(f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}")
        
    edm_diffusion_model.eval()
    latents = torch.randn((2048, 2))
    sampled = edm_diffusion_model.sample(None, None, latents=latents)
    log.info(f"Sampled size: {sampled.shape}")
    