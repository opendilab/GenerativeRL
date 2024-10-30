import torch
import torch.nn as nn
import numpy as np
from grl.neural_network.encoders import GaussianFourierProjectionTimeEncoder


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, num_vars):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.num_vars = num_vars

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.num_vars,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.num_vars,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.num_vars,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralTemporalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralTemporalConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.time_embedding_dim = 256

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.time_embedding_dim,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.time_embedding_dim,
                dtype=torch.cfloat,
            )
        )

        self.time_hidden_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(32, self.time_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.time_hidden_dim, self.time_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.time_hidden_dim, self.time_hidden_dim),
            nn.ReLU(),
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,bioxy->boxy", input, weights)

    def forward(self, t, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        t = self.time_mlp(t)

        # transform t to complex tensor
        t = t.to(torch.cfloat)

        weights1 = torch.einsum("bt,ioxyt->bioxy", t, self.weights1)
        weights2 = torch.einsum("bt,ioxyt->bioxy", t, self.weights2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralTemporalMultivariableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, num_vars):
        super(SpectralTemporalMultivariableConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.num_vars = num_vars
        self.time_embedding_dim = 256

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.num_vars,
                self.modes1,
                self.modes2,
                self.time_embedding_dim,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.num_vars,
                self.modes1,
                self.modes2,
                self.time_embedding_dim,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,biovxy->bovxy", input, weights)

    def forward(self, t, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        weights1 = torch.einsum("bt,iovxyt->biovxy", t, self.weights1)
        weights2 = torch.einsum("bt,iovxyt->biovxy", t, self.weights2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.num_vars,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :, : self.modes1, : self.modes2], weights1
        )
        out_ft[:, :, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :, -self.modes1 :, : self.modes2], weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2dLayer(nn.Module):
    def __init__(self, modes1, modes2, num_vars, width):
        super(FNO2dLayer, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_vars = num_vars

        self.conv = SpectralConv2d(
            self.width, self.width, self.modes1, self.modes2, self.num_vars
        )
        self.w = nn.Conv3d(self.width, self.width, 1)
        self.activation = torch.nn.GELU()

    def forward(self, x):

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FNO2dTemporalLayer(nn.Module):
    def __init__(self, modes1, modes2, in_channels, out_channels):
        super(FNO2dTemporalLayer, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = SpectralTemporalConv2d(
            self.in_channels, self.out_channels, self.modes1, self.modes2
        )
        self.w = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.activation = torch.nn.GELU()

    def forward(self, t, x):

        x1 = self.conv(t, x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FNO2dTemporalMultivariableLayer(nn.Module):
    def __init__(self, modes1, modes2, num_vars, in_channels, out_channels):
        super(FNO2dTemporalMultivariableLayer, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars

        self.conv = SpectralTemporalMultivariableConv2d(
            self.in_channels, self.out_channels, self.modes1, self.modes2, self.num_vars
        )
        self.w = nn.Conv3d(self.in_channels, self.out_channels, 1)
        self.activation = torch.nn.GELU()

    def forward(self, t, x):

        x1 = self.conv(t, x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, num_vars, width, num_layers):
        super(FNO2d, self).__init__()

        """
        2D Fourier Neural Operator
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.num_vars = num_vars

        self.layers = nn.ModuleList(
            [
                FNO2dLayer(self.modes1, self.modes2, self.num_vars, self.width)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class FNO2dTemporal(nn.Module):
    def __init__(self, modes1, modes2, in_channels, out_channels, num_layers):
        super(FNO2dTemporal, self).__init__()

        """
        2D Fourier Neural Operator
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.time_encoder = GaussianFourierProjectionTimeEncoder(embed_dim=32, scale=30)

        self.layers1 = nn.ModuleList(
            [
                FNO2dTemporalLayer(
                    self.modes1, self.modes2, self.in_channels, self.out_channels
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layer1_w = nn.Conv2d(self.in_channels, self.out_channels, 1)

        self.layers2 = nn.ModuleList(
            [
                FNO2dTemporalLayer(
                    self.modes1, self.modes2, self.in_channels, self.out_channels
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layer2_w = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, t, x):
        time_embedding = self.time_encoder(t)

        x0 = x
        for i in range(self.num_layers):
            x = self.layers1[i](time_embedding, x)
        x = x + self.layer1_w(x0)
        # x1 = x
        # for i in range(self.num_layers):
        #    x = self.layers2[i](time_embedding, x)
        # x = x + self.layer2_w(x1)

        return x


class FNO2dTemporalMultivariable(nn.Module):
    def __init__(self, modes1, modes2, num_vars, in_channels, out_channels, num_layers):
        super(FNO2dTemporalMultivariable, self).__init__()

        """
        2D Fourier Neural Operator
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_vars = num_vars
        self.time_encoder = GaussianFourierProjectionTimeEncoder(embed_dim=32, scale=30)
        self.time_hidden_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(32, self.time_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.time_hidden_dim, self.time_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.time_hidden_dim, self.time_hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                FNO2dTemporalMultivariableLayer(
                    self.modes1,
                    self.modes2,
                    self.num_vars,
                    self.in_channels,
                    self.out_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, t, x):
        time_embedding = self.time_encoder(t)
        time_embedding = self.time_mlp(time_embedding)
        for i in range(self.num_layers):
            x = self.layers[i](time_embedding, x)
        return x
