"""
The implementation of GP-VAE for the partially-observed time-series imputation task.

Refer to the paper Fortuin V, Baranchuk D, Rätsch G, et al.
GP-VAE: Deep probabilistic time series imputation. AISTATS. PMLR, 2020: 1651-1661.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>


import numpy as np
import torch
import torch.nn as nn

from .submodules import (
    Encoder,
    rbf_kernel,
    diffusion_kernel,
    matern_kernel,
    cauchy_kernel,
    Decoder,
)


class _GPVAE(nn.Module):
    """model GPVAE with Gaussian Process prior

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel="cauchy",
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
    ):
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(input_dim, latent_dim, encoder_sizes, window_size)
        self.decoder = Decoder(latent_dim, input_dim, decoder_sizes)
        self.M = M
        self.K = K

        self.prior = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if not torch.is_tensor(z):
            z = torch.tensor(z).float()
        num_dim = len(z.shape)
        assert num_dim > 2
        return self.decoder(torch.transpose(z, num_dim - 1, num_dim - 2))

    def forward(self, inputs, training=True):
        x = inputs["X"]
        m_mask = inputs["missing_mask"]
        x = x.repeat(self.M * self.K, 1, 1)

        if self.prior is None:
            self.prior = self._init_prior(device=x.device)

        if m_mask is not None:
            m_mask = m_mask.repeat(self.M * self.K, 1, 1)
            m_mask = m_mask.type(torch.bool)

        # pz = self.prior()
        qz_x = self.encode(x)
        z = qz_x.rsample()
        px_z = self.decode(z)

        nll = -px_z.log_prob(x)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if m_mask is not None:
            nll = torch.where(m_mask, nll, torch.zeros_like(nll))
        nll = nll.sum(dim=(1, 2))

        if self.K > 1:
            kl = qz_x.log_prob(z) - self.prior.log_prob(z)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)

            weights = -nll - kl
            weights = torch.reshape(weights, [self.M, self.K, -1])

            elbo = torch.logsumexp(weights, dim=1)
            elbo = elbo.mean()
        else:
            kl = self.kl_divergence(qz_x, self.prior)
            kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
            kl = kl.sum(1)

            elbo = -nll - self.beta * kl
            elbo = elbo.mean()

        imputed_data = self.decode(self.encode(x).mean).mean * ~m_mask + x * m_mask

        if not training:
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        results = {
            "loss": -elbo.mean(),
            "imputed_data": imputed_data,
        }
        return results

    @staticmethod
    def kl_divergence(a, b):
        return torch.distributions.kl.kl_divergence(a, b)

    def _init_prior(self, device="cpu"):
        # Compute kernel matrices for each latent dimension
        kernel_matrices = []
        for i in range(self.kernel_scales):
            if self.kernel == "rbf":
                kernel_matrices.append(
                    rbf_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "diffusion":
                kernel_matrices.append(
                    diffusion_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "matern":
                kernel_matrices.append(
                    matern_kernel(self.time_length, self.length_scale / 2**i)
                )
            elif self.kernel == "cauchy":
                kernel_matrices.append(
                    cauchy_kernel(
                        self.time_length, self.sigma, self.length_scale / 2**i
                    )
                )

        # Combine kernel matrices for each latent dimension
        tiled_matrices = []
        total = 0
        for i in range(self.kernel_scales):
            if i == self.kernel_scales - 1:
                multiplier = self.latent_dim - total
            else:
                multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                total += multiplier
            tiled_matrices.append(
                torch.unsqueeze(kernel_matrices[i], 0).repeat(multiplier, 1, 1)
            )
        kernel_matrix_tiled = torch.cat(tiled_matrices)
        assert len(kernel_matrix_tiled) == self.latent_dim
        prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim, self.time_length, device=device),
            covariance_matrix=kernel_matrix_tiled.to(device),
        )

        return prior
