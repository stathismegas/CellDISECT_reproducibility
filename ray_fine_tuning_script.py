from dis2p_reproducibility.dis2p import dis2pvi_cE as dvi
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune, air
import warnings
from scipy.sparse import csr_matrix
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import anndata as ad
import scanpy as sc
import os

import scvi

scvi.settings.seed = 0
torch.set_float32_matmul_precision('medium')
warnings.simplefilter("ignore", UserWarning)

DATA_NAME = 'Eraslan'


############################################################################################################
# I'll update this if anything comes to my mind about the parameters or the scripts, will keep you updated
# Please update the code if you see any mistakes or if you have any suggestions
############################################################################################################

def train_fn(config, plan_keys=None, training_keys=None):

    # CHANGE PATH TO YOUR DATA
    # USE ABSOLUTE PATH (STARTING FROM ROOT --> /)
    adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1200.h5ad')
    adata = adata[adata.X.sum(1) != 0].copy()
    sc.pp.subsample(adata, fraction=0.1)

    # CHANGE TO CATEGORIES IN YOUR DATA
    cats = ['tissue', 'Sample ID', 'cell_type', 'sex', 'Age_bin']

    dvi.Dis2pVI_cE.setup_anndata(
        adata,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    model = dvi.Dis2pVI_cE(adata,
                           n_layers=config['n_layers']
                           )

    x_loss = {f'x_{i}_validation': 0 for i in range(len(cats)+1)}
    z_loss = {f'z_{i}_validation': 0 for i in range(len(cats)+1)}
    metrics = {'loss_validation': 'loss_validation',
               'acc_validation': 'acc_validation',
               'f1_validation': 'f1_validation',
               'adv_ce_validation': 'adv_ce_validation',
               'adv_acc_validation': 'adv_acc_validation',
               'adv_f1_validation': 'adv_f1_validation'
               }
    metrics.update(x_loss)
    metrics.update(z_loss)
    callback = TuneReportCallback(metrics=metrics,
                                  on="validation_end")

    training_dict = {key: config[key] for key in training_keys}
    plan_kwargs = {key: config[key] for key in plan_keys}

    model.train(**training_dict,
                plan_kwargs=plan_kwargs,
                callbacks=[callback],
                )
    # model.save(f"{pre_path}/{model_name}")


training_params = {
    'max_epochs': tune.choice([50, 80, 100, 150]),
    'batch_size': tune.choice([512]),
    'cf_weight': tune.loguniform(1e-2, 1e2),
    'beta': tune.loguniform(1e-2, 1e2),
    'clf_weight': tune.loguniform(1e-2, 1e2),
    'adv_clf_weight': tune.loguniform(1e-2, 1e2),
    'adv_period': tune.choice([1, 2, 3]),
    'n_cf': tune.choice([1, 2, 3]),
}

plan_kwargs = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
}


config = {'n_layers': tune.choice([1, 2, 3]),
          'n_hidden': tune.choice([128, 256, 512, 1024]),
          'n_latent_shared': tune.choice([10, 20, 30, 40, 50]),
          'n_latent_attribute': tune.choice([10, 20, 30, 40, 50]),
          'dropout_rate': tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
          }
config.update(training_params)
config.update(plan_kwargs)


scheduler = ASHAScheduler(
    max_t=500,
    grace_period=5,
    reduction_factor=4)

metrics = {'loss_validation': 'loss_validation',
           'acc_validation': 'acc_validation',
           'f1_validation': 'f1_validation',
           'adv_ce_validation': 'adv_ce_validation',
           'adv_acc_validation': 'adv_acc_validation',
           'adv_f1_validation': 'adv_f1_validation'
           }

reporter = CLIReporter(
    parameter_columns=list(plan_kwargs.keys()) + list(training_params.keys()),
    metric_columns=list(metrics.keys())
)
plan_keys = plan_kwargs.keys()
training_keys = training_params.keys()

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(
            train_fn,
            plan_keys=plan_keys,
            training_keys=training_keys,),
        resources={
            "cpu": 4,
            "gpu": 0.2
        }
    ),
    tune_config=tune.TuneConfig(
        metric="loss_validation",
        mode="min",
        scheduler=scheduler,
        num_samples=500,
    ),
    run_config=air.RunConfig(
        name=f"tune_dis2p_{DATA_NAME}",
        progress_reporter=reporter,
        storage_path='/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2p/', # CHANGE TO YOUR PATH
        log_to_file=True,
    ),
    param_space=config,
)
results = tuner.fit()

print("Best hyperparameters found were: ", results.get_best_result().config)
