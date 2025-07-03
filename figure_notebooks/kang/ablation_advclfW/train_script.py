import sys
adv_clf_weight = float(sys.argv[1])
split_key = sys.argv[2]

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

scvi.settings.seed = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad')
adata.X = adata.layers['counts'].copy()
adata = adata[adata.X.sum(1) != 0].copy()

cats = ['cell_type', 'condition']

arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 350,
 'batch_size': 128,
 'recon_weight': 1,
 'cf_weight': 1,
 'beta': 1,
 'clf_weight': 1,
 'adv_clf_weight': adv_clf_weight,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 15,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
}

plan_kwargs = {
 'lr': 0.001,
 'weight_decay': 0.001,
 'new_cf_method': True,
 'lr_patience': 5,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


module_name = f'kang_ablation_celldisect_advclfW_all1'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)


# specify a name for your model
model_name =  f'celldisect_{split_key}_advclfW_{adv_clf_weight}'
wandb_logger = WandbLogger(project=f"CellDISECT_{split_key}_AblationAdvCLF_Kang", name=model_name)
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
    continuous_covariate_keys=[]
)
model = CellDISECT(adata,
                       split_key=split_key,
                       train_split=['train'],
                       valid_split=['valid'],
                       test_split=['ood'],
                       **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
