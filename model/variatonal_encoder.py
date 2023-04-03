import math

import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, color_channels, model_size):
        super().__init__()
        in_channel = color_channels
        # convolutional encoder
        if model_size == "small":
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channel, 32, 3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(True),
            )
            # create the mu and log var layers
            self.encoder_lin_mu = nn.Linear(7 * 7 * 64, encoded_space_dim)
            self.encoder_lin_var = nn.Linear(7 * 7 * 64, encoded_space_dim)
        else:
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channel, 32, 3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            # create the mu and log var layers
            self.encoder_lin_mu = nn.Linear(4 * 4 * 256, encoded_space_dim)
            self.encoder_lin_var = nn.Linear(4 * 4 * 256, encoded_space_dim)
        # flattening layer
        self.flatten = nn.Flatten(start_dim=1)
        # relu layer
        self.relu = nn.ReLU(True)



    def forward(self, x):
        # pass through the encoder
        x = self.encoder_cnn(x)
        # flatten the output
        x = self.flatten(x)
        # get the mu and log var
        mu = self.encoder_lin_mu(x)
        log_var = self.encoder_lin_var(x)
        # perform reparameterization trick to sample from the latent space
        z = reparameterize(mu, log_var)
        return z, mu, log_var

    def calc_mi(self, x):
        """
        adjusted from (https://github.com/jxhe/vae-lagging-encoder)
        Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """
        # [x_batch, nz]
        _, mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = reparameterize(mu, logvar)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()


def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)


def log_sum_exp(value, dim=None, keepdim=False):
    """
    adjusted from (https://github.com/jxhe/vae-lagging-encoder)
    Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)
