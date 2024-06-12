# Test grl/generative_models/diffusion_model/diffusion_model.py

import os
import signal
import sys
import unittest

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from matplotlib import animation
from rich.progress import track
from sklearn.datasets import make_swiss_roll

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils import set_seed
from grl.utils.log import log


class TestDiffusionModel(unittest.TestCase):

    def test_unconditioned_diffusion_model(self):

        x_size = 2
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        t_embedding_dim = 32
        config = EasyDict(
            dict(
                device=device,
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
                            t_encoder=dict(
                                type="GaussianFourierProjectionTimeEncoder",
                                args=dict(
                                    embed_dim=t_embedding_dim,
                                    scale=30.0,
                                ),
                            ),
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
                    data_num=100,
                    iterations=1000,
                    batch_size=16,
                    clip_grad_norm=1.0,
                    eval_freq=500,
                    device=device,
                ),
            )
        )

        diffusion_model = DiffusionModel(config=config.diffusion_model).to(
            config.diffusion_model.device
        )
        # if python version is 3.12 or higher, torch.compile is not available
        if sys.version_info[1] < 12:
            diffusion_model = torch.compile(diffusion_model)

        # get data
        data = make_swiss_roll(n_samples=config.parameter.data_num, noise=0.01)[
            0
        ].astype(np.float32)[:, [0, 2]]
        # transform data
        data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
        data[:, 1] = data[:, 1] / np.max(np.abs(data[:, 1]))
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 10 - 5

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=config.parameter.batch_size, shuffle=True
        )

        def get_train_data(dataloader):
            while True:
                yield from dataloader

        data_generator = get_train_data(data_loader)

        batch_data = next(data_generator)
        batch_data = batch_data.to(config.device)

        diffusion_model.train()
        loss = diffusion_model.flow_matching_loss(x=batch_data)
        loss = diffusion_model.score_matching_loss(x=batch_data)

        diffusion_model.eval()
        t_span = torch.linspace(0.0, 1.0, 5)

        x_t = diffusion_model.sample_forward_process(t_span=t_span)
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=20)
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process(t_span=t_span, x_0=batch_data)
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process(
            t_span=t_span, batch_size=20, x_0=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample(t_span=t_span)
        assert x_t.shape[0] == 2

        x_t = diffusion_model.sample(t_span=t_span, batch_size=20)
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample(t_span=t_span, x_0=batch_data)
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample(t_span=t_span, batch_size=20, x_0=batch_data)
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        fixed_x = (
            torch.tensor([1.0, 0.0])
            .to(config.device)
            .unsqueeze(0)
            .expand(batch_data.shape[0], -1)
        )
        fixed_mask = (
            torch.tensor([0.0, 1.0])
            .to(config.device)
            .unsqueeze(0)
            .expand(batch_data.shape[0], -1)
        )

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, batch_size=20
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, x_0=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            x_0=batch_data,
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span
        )
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, batch_size=20
        )
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, x_0=batch_data
        )
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            x_0=batch_data,
        )
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

    def test_conditional_diffusion_model(self):

        x_size = 2
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        t_embedding_dim = 32
        config = EasyDict(
            dict(
                device=device,
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
                            t_encoder=dict(
                                type="GaussianFourierProjectionTimeEncoder",
                                args=dict(
                                    embed_dim=t_embedding_dim,
                                    scale=30.0,
                                ),
                            ),
                            backbone=dict(
                                type="TemporalSpatialResidualNet",
                                args=dict(
                                    hidden_sizes=[512, 256, 128],
                                    output_dim=x_size,
                                    t_dim=t_embedding_dim,
                                    condition_dim=x_size,
                                    condition_hidden_dim=32,
                                    t_condition_hidden_dim=128,
                                ),
                            ),
                        ),
                    ),
                ),
                parameter=dict(
                    training_loss_type="score_matching",
                    lr=5e-3,
                    data_num=100,
                    iterations=1000,
                    batch_size=16,
                    clip_grad_norm=1.0,
                    eval_freq=500,
                    device=device,
                ),
            )
        )

        diffusion_model = DiffusionModel(config=config.diffusion_model).to(
            config.diffusion_model.device
        )
        # if python version is 3.12 or higher, torch.compile is not available
        if sys.version_info[1] < 12:
            diffusion_model = torch.compile(diffusion_model)

        # get data
        data = make_swiss_roll(n_samples=config.parameter.data_num, noise=0.01)[
            0
        ].astype(np.float32)[:, [0, 2]]
        # transform data
        data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
        data[:, 1] = data[:, 1] / np.max(np.abs(data[:, 1]))
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 10 - 5

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=config.parameter.batch_size, shuffle=True
        )

        def get_train_data(dataloader):
            while True:
                yield from dataloader

        data_generator = get_train_data(data_loader)

        batch_data = next(data_generator)
        batch_data = batch_data.to(config.device)

        diffusion_model.train()
        loss = diffusion_model.flow_matching_loss(x=batch_data, condition=batch_data)
        loss = diffusion_model.score_matching_loss(x=batch_data, condition=batch_data)

        diffusion_model.eval()
        t_span = torch.linspace(0.0, 1.0, 5)

        x_t = diffusion_model.sample_forward_process(
            t_span=t_span, condition=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process(
            t_span=t_span, batch_size=20, condition=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample_forward_process(
            t_span=t_span, x_0=batch_data, condition=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process(
            t_span=t_span, batch_size=20, x_0=batch_data, condition=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample(t_span=t_span, condition=batch_data)
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample(t_span=t_span, batch_size=20, condition=batch_data)
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample(
            t_span=t_span, x_0=batch_data, condition=batch_data
        )
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample(
            t_span=t_span, batch_size=20, x_0=batch_data, condition=batch_data
        )
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        fixed_x = (
            torch.tensor([1.0, 0.0])
            .to(config.device)
            .unsqueeze(0)
            .expand(batch_data.shape[0], -1)
        )
        fixed_mask = (
            torch.tensor([0.0, 1.0])
            .to(config.device)
            .unsqueeze(0)
            .expand(batch_data.shape[0], -1)
        )

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, condition=batch_data
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            condition=batch_data,
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            x_0=batch_data,
            condition=batch_data,
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            x_0=batch_data,
            condition=batch_data,
        )
        assert x_t.shape[0] == 5
        assert x_t.shape[1] == 20
        assert x_t.shape[2] == batch_data.shape[0]
        assert x_t.shape[3] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x, fixed_mask=fixed_mask, t_span=t_span, condition=batch_data
        )
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            condition=batch_data,
        )
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            x_0=batch_data,
            condition=batch_data,
        )
        assert x_t.shape[0] == batch_data.shape[0]
        assert x_t.shape[1] == 2

        x_t = diffusion_model.sample_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=20,
            x_0=batch_data,
            condition=batch_data,
        )
        assert x_t.shape[0] == 20
        assert x_t.shape[1] == batch_data.shape[0]
        assert x_t.shape[2] == 2


if __name__ == "__main__":
    test_class = TestDiffusionModel()
    test_class.test_unconditioned_diffusion_model()
    test_class.test_conditional_diffusion_model()
