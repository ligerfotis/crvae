import torch
from torch import nn

from model.decoder import Decoder
from utils import recon_loss
from model.variatonal_encoder import Encoder


class CRVAE(nn.Module):
    def __init__(self, z_dim=128, channels=3, beta=1, gamma=1, K=4096, m=0.99, T=0.1, device=None, loss_type='mse'):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.encoder = Encoder(encoded_space_dim=z_dim, color_channels=channels)
        self.encoder_target = Encoder(encoded_space_dim=z_dim, color_channels=channels)
        self.decoder = Decoder(encoded_space_dim=z_dim, color_channels=channels)

        self.beta = beta
        self.gamma = gamma
        # initialize the key encoder to be the same as the query encoder
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue for the keys and the pointer for the queue
        self.register_buffer("queue", torch.randn(z_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # initialize the reconstruction loss
        self.reconstruction_loss = recon_loss(loss_type)

    def encode(self, im1, im2):
        # encode
        z, mu, var = self.encoder(im1)
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        # compute symmetric loss
        con_loss, q, k, labels = self.contrastive_loss(im1, im2)
        self._dequeue_and_enqueue(k)
        return con_loss, z, q, mu, var, labels, k

    def forward(self, im1, im2=None, inference=False):
        if not inference:
            con_loss, z, q, mu, var, labels, k = self.encode(im1, im2)
        else:
            z, mu, var = self.encoder(im1)
            q = z
            k = z
            con_loss = 0

        # get the reconstruction x_hat
        x_hat = self.decoder(z)
        return x_hat, z, mu, var, q, k, con_loss

    def get_kl(self):
        return self.encoder.kl

    def reparameterize(mu, log_var, sampling_distribution):
        # sample noise from a normal distribution
        epsilon = sampling_distribution.sample(mu.shape)
        # get the std
        sigma = torch.exp(log_var / 2)
        # re-parameterization trick
        z = mu + epsilon * sigma
        return z

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute key features
        z_q, mu_q, var_q = self.encoder(im_q)  # queries: NxC
        q = z_q
        q = nn.functional.normalize(q, dim=1)
        # compute query features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            z_k, mu_k, var_k = self.encoder_target(im_k_)  # keys: NxC
            k = z_k
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        # print(logits.shape)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k, labels

    def predict_batch(self, batch):
        """
        This function is used to predict the latent space of a batch of images.
        """
        img_1 = batch.cuda(non_blocking=True)
        with torch.no_grad():
            reconstructed, z, mu, var, q, k, con_loss = self.forward(img_1, inference=True)

        return reconstructed, mu, var, z
