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


class FFVAE(BaseModuleClass):
    """Flexibly Fair Variational auto-encoder module.

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
                            return_dist=True,
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

        self.s_classifiers_list = nn.ModuleList([])
        for i in range(self.zs_num):
            self.s_classifiers_list.append(
                Classifier(
                    n_input=1,
                    n_labels=self.n_cat_list[i],
                ).to(device)
            )

    def set_require_grad(self, mode):
        if TRAIN_MODE.CLASSIFICATION not in mode:
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = False
        else:
            for classifier in self.s_classifiers_list:
                classifier.requires_grad = True

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
    def inference(self, x,):
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz, z = self.z_encoder(x_)

        outputs = {"z": z, "qz": qz, "library": library}
        return outputs

    @auto_move_data
    def generative(self, z, library,):

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

    def classification_logits(self, inference_outputs):
        z = inference_outputs["z"]
        z_split = torch.split(z, 1, dim=1)
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(z_split[i])
            logits += [logits_i]

        return logits

    def compute_clf_metrics(self, logits, cat_covs):
        # CE, ACC, F1
        cats = torch.split(cat_covs, 1, dim=1)
        ce_losses = []
        accuracy_scores = []
        f1_scores = []
        for i in range(self.zs_num):
            s_i = one_hot_cat([self.n_cat_list[i]], cats[i]).to(device)
            ce_losses += [F.cross_entropy(logits[i], s_i)]
            kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
            predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
            acc = Accuracy(**kwargs).to(device)
            accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
            F1 = F1Score(**kwargs).to(device)
            f1_scores.append(F1(predicted_labels, cats[i]).to(device))

        ce_loss_sum = sum(torch.mean(ce) for ce in ce_losses)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)
        f1 = sum(f1_scores) / len(f1_scores)

        return ce_loss_sum, accuracy, f1

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            beta: Tunable[Union[float, int]],  # KL Z weight
            clf_weight: Tunable[Union[float, int]],  # Si classifier weight
            kl_weight: float = 1.0,
    ):
        # reconstruction loss X

        x = tensors[REGISTRY_KEYS.X_KEY]
        reconst_loss_x = -torch.mean(generative_outputs["px"].log_prob(x).sum(-1))

        # KL divergence Z

        z = inference_outputs["z"]
        qz_prior = Normal(torch.zeros_like(z), torch.ones_like(z))

        kld = torch.mean(kl(inference_outputs["qz"], qz_prior).sum(dim=1))

        # classification metrics: CE, ACC, F1

        cat_covs = tensors[REGISTRY_KEYS.CAT_COVS_KEY]
        logits = self.classification_logits(inference_outputs)
        ce_loss_sum, accuracy, f1 = self.compute_clf_metrics(logits, cat_covs)
        ce_loss_mean = ce_loss_sum / len(range(self.zs_num))

        # total loss
        loss = reconst_loss_x + \
               kld * kl_weight * beta + \
               ce_loss_sum * clf_weight

        loss_dict = {
            LOSS_KEYS.LOSS: loss,
            LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x,
            LOSS_KEYS.KL_Z: kld,
            LOSS_KEYS.CLASSIFICATION_LOSS: ce_loss_mean,
            LOSS_KEYS.ACCURACY: accuracy,
            LOSS_KEYS.F1: f1
        }

        return loss_dict
