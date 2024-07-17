import sys
recon_weight = float(sys.argv[1])
cf_weight = float(sys.argv[2])

import os
import shutil

import scvi
import scanpy as sc
import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from dis2p import dis2pvi_cE as dvi

scvi.settings.seed = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']
split_key = 'split_1'

module_name = f'dis2p_cE_{split_key}_cf_generalization'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)

arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 100,
 'batch_size': 256,
 'recon_weight': recon_weight,
 'cf_weight': cf_weight,
 'beta': 0.003,
 'clf_weight': 0.8,
 'adv_clf_weight': 0.015,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 15,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
 'n_epochs_pretrain_ae': 10,
}

plan_kwargs = {
 'lr': 0.01,
 'weight_decay': 0.0005,
 'new_cf_method': True,
 'lr_patience': 6,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


# specify a name for your model
model_name = (
    f'pretrainAE_{train_dict["n_epochs_pretrain_ae"]}_'
    f'maxEpochs_{train_dict["max_epochs"]}_split_{split_key}_'
    f'reconW_{train_dict["recon_weight"]}_'
    f'cfWeight_{train_dict["cf_weight"]}_'
    f'beta_{train_dict["beta"]}_'
    f'clf_{train_dict["clf_weight"]}_'
    f'adv_{train_dict["adv_clf_weight"]}_'
    f'advp_{train_dict["adv_period"]}_'
    f'n_cf_{train_dict["n_cf"]}_'
    f'lr_{plan_kwargs["lr"]}_'
    f'wd_{plan_kwargs["weight_decay"]}_'
    f'new_cf_{plan_kwargs["new_cf_method"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}'
)

wandb_logger = WandbLogger(project=f"Dis2PVI_cE_{split_key}_cf_generalization", name=model_name)
train_dict['logger'] = wandb_logger
wandb_logger.experiment.config.update({'train_dict': train_dict, 'arch_dict': arch_dict, 'plan_kwargs': plan_kwargs})
try: # Clean up the directory if it exists, overwrite the model
    shutil.rmtree(f"{pre_path}/{model_name}")
    print("Directory deleted successfully")
except OSError as e:
    print(f"Error deleting directory: {e}") 

dvi.Dis2pVI_cE.setup_anndata(
    adata,
    layer='counts',
    categorical_covariate_keys=cats,
    continuous_covariate_keys=[]
)
model = dvi.Dis2pVI_cE(adata,
                       split_key=split_key,
                       train_split=['train'],
                       valid_split=['val'],
                       test_split=['test'],
                       **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
