import sys
adv_clf_weight = float(sys.argv[1])


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

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad')
adata.X = adata.layers['counts'].copy()
adata = adata[adata.X.sum(1) != 0].copy()

cats = ['cell_type', 'condition']
split_key = 'split_CD4 T'

arch_dict = {'n_layers': 2,
 'n_hidden': 1024,
 'n_latent_shared': 128,
 'n_latent_attribute': 128,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 350,
 'batch_size': 64,
 'recon_weight': 10,
 'cf_weight': 0.5,
 'beta': 0.0029,
 'clf_weight': 0.4,
 'adv_clf_weight': adv_clf_weight,
 'adv_period': 4,
 'n_cf': 1,
 'early_stopping_patience': 25,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
}

plan_kwargs = {
 'lr': 0.001,
 'weight_decay': 0.001,
 'new_cf_method': True,
 'lr_patience': 5,
 'lr_factor': 0.9,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


module_name = f'kang_ablation_dis2p_advclfW'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)


# specify a name for your model
model_name =  f'dis2p_cE_{split_key}_advclfW_{adv_clf_weight}'
wandb_logger = WandbLogger(project=f"Dis2PVI_cE_{split_key}_AblationAdvCLF_Kang", name=model_name)
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
                       valid_split=['valid'],
                       test_split=['ood'],
                       **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
