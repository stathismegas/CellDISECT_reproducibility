import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from scvi.module import VAE
from scvi.module.base import BaseModuleClass
from scvi.dataloaders._data_splitting import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.model.base import RNASeqMixin, VAEMixin, BaseModelClass
from scvi.train import TrainingPlan, TrainRunner

from .TCtrainingplan import TCTrainingPlan
from .factorvae import FactorVAE
from .ffvae import FFVAE
from .dipvae import DIPVAE
from .betaTCvae import BetaTCVAE

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from scvi.autotune._types import Tunable, TunableMixin

from .utils import *


class VI(
    RNASeqMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
    TunableMixin
):
    """
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> VI.setup_anndata(adata, batch_key="batch")
    >>> vae = VI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    """

    _data_splitter_cls = DataSplitter
    _train_runner_cls = TrainRunner

    def __init__(
            self,
            adata: AnnData,
            module_id: int,  # (MODULE_ID ENUM)
            n_hidden: int = 128,
            n_latent: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            **model_kwargs,
    ):
        super().__init__(adata)

        self._training_plan_cls = TCTrainingPlan

        module_id_to_cls = {MODULE_ID.FACTOR_VAE: FactorVAE,
                            MODULE_ID.FF_VAE: FFVAE,
                            MODULE_ID.BETA_TC_VAE: BetaTCVAE,
                            MODULE_ID.DIP_VAE: DIPVAE}

        self._module_cls = module_id_to_cls[module_id]

        self._data_loader_cls = AnnDataLoader

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VI Model with the following params: \nn_hidden: {}, n_latent: {}"
            ", n_layers: {}, dropout_rate: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    def is_adversarial(self):
        return self._module_cls == FactorVAE or self._module_cls == FFVAE

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: Optional[str] = None,
            batch_key: Optional[str] = None,
            labels_key: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """%(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.
        batch_size
            Batch size to use.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            latent += [outputs["z"].cpu()]

        return torch.cat(latent).numpy()

    # @devices_dsp.dedent
    def train(
            self,
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = True,
            train_size: float = 0.8,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = True,
            plan_kwargs: Optional[dict] = None,
            beta: Tunable[Union[float, int]] = 1,  # KL Zi weight
            clf_weight: Tunable[Union[float, int]] = 50,  # Si classifier weight
            tc_weight: Tunable[Union[float, int]] = 10,  # TC weight
            **trainer_kwargs,
    ):
        """Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_use_gpu)s
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        beta
            KL Zi weight
        clf_weight
            Si classifier weight
        tc_weight
            TC weight
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = DataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=True,
            batch_size=batch_size,
        )
        plan_kwargs.update({'beta': beta})
        if self._module_cls == BetaTCVAE:
            plan_kwargs['tc_weight'] = tc_weight
        if self._module_cls == FFVAE:
            plan_kwargs['clf_weight'] = clf_weight
        training_plan = self._training_plan_cls(self.module,
                                                adversarial_classifier=self.is_adversarial(),
                                                **plan_kwargs)

        training_plan.tc_weight = tc_weight

        if self._module_cls == BetaTCVAE:
            training_plan.is_betaTC = True

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs['early_stopping_monitor'] = "loss_validation"
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
