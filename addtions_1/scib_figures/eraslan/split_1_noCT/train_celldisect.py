from celldisect_params import train_dict, arch_dict, plan_kwargs

import os
import shutil
import sys

latent_dim = int(sys.argv[1])
arch_dict['n_latent_shared'] = latent_dim
arch_dict['n_latent_attribute'] = latent_dim

import scvi
import scanpy as sc
import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from celldisect import CellDISECT

scvi.settings.seed = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/lotfollahi/aa34/celldisect/datasets/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

cats = ['tissue', 'Sample ID', 'Age_bin']
split_key = 'split_1'

module_name = f'celldisect_{split_key}'
pre_path = f'/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/models_celldisect/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)

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
    f'new_cf_{plan_kwargs["ensemble_method_cf"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}'
    f'batch_size_{train_dict["batch_size"]}_'
    f'NoCT'
)

wandb_logger = WandbLogger(project=f"CellDISECT_NoCT_{split_key}", name=model_name, entity='dis2p')
train_dict['logger'] = wandb_logger
wandb_logger.experiment.config.update({'train_dict': train_dict, 'arch_dict': arch_dict, 'plan_kwargs': plan_kwargs})
try: # Clean up the directory if it exists, overwrite the model
    shutil.rmtree(f"{pre_path}/{model_name}")
    print("Directory deleted successfully")
except OSError as e:
    print(f"Error deleting directory: {e}") 

CellDISECT.setup_anndata(
    adata,
    layer='counts',
    categorical_covariate_keys=cats,
    continuous_covariate_keys=[],
    add_cluster_covariate=True,
)
model = CellDISECT(adata,
                       split_key=split_key,
                       train_split=['train'],
                       valid_split=['val'],
                       test_split=['test'],
                       **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
