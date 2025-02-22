import os
import shutil

import scvi
import scanpy as sc
import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from celldisect import CellDISECT

scvi.settings.seed = 420
print(torch.cuda.is_available())

data_path = 'path/to/data.h5ad'

adata = sc.read(data_path)
adata = adata[adata.X.sum(1) != 0].copy()
adata.var['highly_variable'] = adata.var['highly_variable'].astype(bool)

cats = ['integration_donor', 'integration_library_platform_coarse',
        'organ']
cell_type_included = False # Set to True if you have provided a cell type annotation in the cats list


arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.1,
 'weighted_classifier': False,
}
train_dict = {
 'max_epochs': 1000,
 'batch_size': 256,
 'recon_weight': 20,
 'cf_weight': 0.8,
 'beta': 0.003,
 'clf_weight': 0.02,
 'adv_clf_weight': 0.014,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 6,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
 'n_epochs_pretrain_ae': 0,
}

plan_kwargs = {
 'lr': 0.003,
 'weight_decay': 0.00005,
 'ensemble_method_cf': True,
 'lr_patience': 5,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


module_name = f'suoFull_NoAge'
pre_path = f'path/to/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)


# specify a name for your model
model_name = (
    f'pretrainAE_{train_dict["n_epochs_pretrain_ae"]}_'
    f'maxEpochs_{train_dict["max_epochs"]}_'
    f'reconW_{train_dict["recon_weight"]}_'
    f'cfWeight_{train_dict["cf_weight"]}_'
    f'beta_{train_dict["beta"]}_'
    f'clf_{train_dict["clf_weight"]}_'
    f'adv_{train_dict["adv_clf_weight"]}_'
    f'advp_{train_dict["adv_period"]}_'
    f'n_cf_{train_dict["n_cf"]}_'
    f'lr_{plan_kwargs["lr"]}_'
    f'wd_{plan_kwargs["weight_decay"]}_'
    f'ensemble_cf_{plan_kwargs["ensemble_method_cf"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}_'
    f'batch_size_{train_dict["batch_size"]}_'
)
if cell_type_included:
    model_name = model_name + f'cellTypeIncluded'
else:
    model_name = model_name + f'cellTypeNotIncluded'

wandb_logger = WandbLogger(project=f"CellDISECT_suoFullNoAge", name=model_name)
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
    add_cluster_covariate=not cell_type_included, # add_cluster_covariate if cell type is not included
)
model = CellDISECT(adata,
                   **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
