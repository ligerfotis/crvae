import torch
from torch import nn

from model.decoder import Decoder
from utils import recon_loss
from model.variatonal_encoder import Encoder


def reparameterize(mu, log_var, sampling_distribution):
    # sample noise from a normal distribution
    epsilon = sampling_distribution.sample(mu.shape)
    # get the std
    sigma = torch.exp(log_var / 2)
    # re-parameterization trick
    z = mu + epsilon * sigma
    return z


class VAE(nn.Module):
    def __init__(self, z_dim=128, channels=3, beta=1, device=None, loss_type='mse', model_size='small', output_size=28):
        super().__init__()
        self.device = device
        self.encoder = Encoder(encoded_space_dim=z_dim, color_channels=channels, model_size=model_size)
        self.decoder = Decoder(encoded_space_dim=z_dim, color_channels=channels, output_size=output_size)

        self.beta = beta

        # initialize the reconstruction loss
        self.reconstruction_loss = recon_loss(loss_type)

    def forward(self, img_org=None):
        z, mu, var = self.encoder(img_org)
        # get the reconstruction x_hat
        x_hat = self.decoder(z)
        return x_hat, z, mu, var

    def get_kl(self):
        return self.encoder.kl

    def predict_batch(self, batch):
        """
        This function is used to predict the latent space of a batch of images.
        """
        img_1 = batch.cuda(non_blocking=True)
        with torch.no_grad():
            reconstructed, z, mu, var = self.forward(img_1)

        return reconstructed, mu, var, z
