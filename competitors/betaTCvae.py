from typing import Callable, Iterable, Literal, Optional, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import logsumexp
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Decoder, Encoder

from torchmetrics import Accuracy, F1Score

torch.backends.cudnn.benchmark = True

from .utils import *

from scvi.module._classifier import Classifier

dim_indices = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from scvi.autotune._types import Tunable, TunableMixin


class BetaTCVAE(BaseModuleClass):
    """beta-TC Variational auto-encoder module.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: Tunable[int] = 128,
            n_latent: Tunable[int] = 10,
            n_layers: Tunable[int] = 1,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input)).to(device)

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders

        n_input_encoder = n_input
        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        self.zs_num = len(self.n_cat_list)

        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=False,
        ).to(device)

        # Decoders

        self.x_decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
        ).to(device)

    def _get_inference_input(self, tensors):

        x = tensors[REGISTRY_KEYS.X_KEY].to(device)

        input_dict = {
            "x": x,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):

        z = inference_outputs["z"]
        library = inference_outputs["library"]

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, ):
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_)
        qz = Normal(qz_m, qz_v.sqrt())

        outputs = {"z": z, "qz_m": qz_m, "qz_v": qz_v, "qz": qz, "library": library}
        return outputs

    @auto_move_data
    def generative(self, z, library, ):

        output_dict = {}

        px_scale, px_r, px_rate, px_dropout = self.x_decoder(
            self.dispersion,
            z,
            library,
        )
        px_r = torch.exp(self.px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        output_dict["px"] = px

        return output_dict

    def log_density_gaussian(self, z: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and var at z
        """
        qz = Normal(mu, var.sqrt())
        return qz.log_prob(z)

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            beta: Tunable[Union[float, int]],  # KL Zi weight
            tc_weight: Tunable[Union[float, int]],  # TC weight
            datasize: Union[float, int],
            kl_weight: float = 1.0,
    ):

        x = tensors[REGISTRY_KEYS.X_KEY]
        reconst_loss_x = -torch.mean(generative_outputs["px"].log_prob(x).sum(-1))

        mu = inference_outputs['qz_m']
        var = inference_outputs['qz_v']
        z = inference_outputs['z']

        log_q_zx = self.log_density_gaussian(z, mu, var).sum(dim=1)

        log_p_z = self.log_density_gaussian(z, torch.zeros_like(z), torch.ones_like(z)).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                var.view(1, batch_size, latent_dim))

        dataset_size = datasize
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        loss = reconst_loss_x + \
               mi_loss + \
               tc_loss * tc_weight + \
               kld_loss * kl_weight * beta

        return {LOSS_KEYS.LOSS: loss,
                LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x,
                LOSS_KEYS.KL_Z: kld_loss,
                'TC_Loss': tc_loss,
                'MI_Loss': mi_loss}
