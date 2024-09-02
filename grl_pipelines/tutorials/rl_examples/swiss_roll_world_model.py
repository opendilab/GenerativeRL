################################################################################################
# This script demonstrates how to use an Independent Conditional Flow Matching (ICFM), which is a flow model, to train a world model by using Swiss Roll dataset.
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

from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.generative_models.metric import compute_likelihood
from grl.rl_modules.world_model.state_prior_dynamic_model import ActionConditionedWorldModel
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
data_num=1000000
config = EasyDict(
    dict(
        device=device,
        dataset=dict(
            data_num=data_num,
            noise=0.6,
        ),
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
                    condition_encoder=t_encoder,
                    backbone=dict(
                        type="TemporalSpatialResidualNet",
                        args=dict(
                            hidden_sizes=[512, 256, 128],
                            output_dim=x_size,
                            t_dim=t_embedding_dim,
                            condition_dim=t_embedding_dim,
                            condition_hidden_dim=64,
                            t_condition_hidden_dim=128,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=5e-4,
            data_num=data_num,
            iterations=100000,
            batch_size=40960,
            clip_grad_norm=1.0,
            eval_freq=500,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint-swiss-roll-icfm-world-model",
            video_save_path="./video-swiss-roll-icfm-world-model",
            device=device,
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    flow_model = IndependentConditionalFlowModel(config=config.flow_model).to(
        config.flow_model.device
    )
    flow_model = torch.compile(flow_model)

    def get_data(data_num):
        # get data
        x_and_t = make_swiss_roll(
            n_samples=data_num, noise=config.dataset.noise
        )
        t = x_and_t[1].astype(np.float32)
        t = (t - np.min(t)) / (np.max(t) - np.min(t))
        x = x_and_t[0].astype(np.float32)[:, [0, 2]]
        # transform data
        x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
        x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 10 - 5

        x0 = x[:-1]
        x1 = x[1:]
        action = t[1:] - t[:-1]
        return x0, x1, action
    
    x0, x1, action = get_data(config.dataset.data_num)

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

    # zip x0, x1, action
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x0).to(config.device),
            torch.tensor(x1).to(config.device),
            torch.tensor(action).to(config.device),
        ),
        batch_size=config.parameter.batch_size,
        shuffle=True,
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

    def render_3d_trajectory_video(data, video_save_path, iteration, fps=100, dpi=100):

        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
            
        T, B, _ = data.shape
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set the axes limits
        ax.set_xlim(np.min(data[:,:,0]), np.max(data[:,:,0]))
        ax.set_ylim(np.min(data[:,:,1]), np.max(data[:,:,1]))
        ax.set_zlim(0, T)

        # Initialize a list of line objects for each point with alpha transparency
        lines = [ax.plot([], [], [], alpha=0.5)[0] for _ in range(B)]

        # Initialization function to set the background of each frame
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        # Animation function which updates each frame
        def update(frame):
            for i, line in enumerate(lines):
                x_data = data[:frame+1, i, 0]
                y_data = data[:frame+1, i, 1]
                z_data = np.arange(frame+1)
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            return lines

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=True)

        # Save the animation
        video_filename = os.path.join(video_save_path, f"iteration_3D_{iteration}.mp4")
        ani.save(video_filename, fps=fps, dpi=dpi)

        # Clean up
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

        #if iteration > 0 and iteration % config.parameter.eval_freq == 0:
        if True:
            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x0_eval, x1_eval, action_eval = get_data(500)
            x0_eval = torch.tensor(x0_eval).to(config.device)
            x1_eval = torch.tensor(x1_eval).to(config.device)
            action_eval = torch.tensor(action_eval).to(config.device)
            action_eval = -torch.ones_like(action_eval).to(config.device)*0.05
            x_t = (
                flow_model.sample_forward_process(t_span=t_span, x_0=x0_eval, condition=action_eval)
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)

        batch_data = next(data_generator)

        flow_model.train()
        if config.parameter.training_loss_type == "flow_matching":
            loss = flow_model.flow_matching_loss(x0=batch_data[0], x1=batch_data[1], condition=batch_data[2])
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

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(flow_model, optimizer, iteration)
