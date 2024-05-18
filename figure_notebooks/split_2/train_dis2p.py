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
adata.X = adata.layers['counts'].copy()
adata = adata[adata.X.sum(1) != 0].copy()

cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']
split_key = 'split_2'

arch_dict = {'n_layers': 2,
 'n_hidden': 2048,
 'n_latent_shared': 512,
 'n_latent_attribute': 512,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 350,
 'batch_size': 1024,
 'recon_weight': 10,
 'cf_weight': 0.5,
 'beta': 0.0029,
 'clf_weight': 0.4,
 'adv_clf_weight': 0.2,
 'adv_period': 3,
 'n_cf': 2,
 'early_stopping_patience': 5,
 'early_stopping': True,
 'save_best': True,
}

plan_kwargs = {
 'lr': 0.003,
 'weight_decay': 0.00006,
 'new_cf_method': True,
 'lr_patience': 2,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
}


module_name = f'dis2p_cE_{split_key}'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)


# specify a name for your model
model_name =  f'weightedCLF_split_{split_key}_cfWeight_{train_dict["cf_weight"]}_beta_{train_dict["beta"]}_clf_{train_dict["clf_weight"]}_adv_{train_dict["adv_clf_weight"]}_advp_{train_dict["adv_period"]}_n_cf_{train_dict["n_cf"]}_lr_{plan_kwargs["lr"]}_wd_{plan_kwargs["weight_decay"]}_new_cf_{plan_kwargs["new_cf_method"]}_dropout_{arch_dict["dropout_rate"]}_n_hidden_{arch_dict["n_hidden"]}_n_latent_{arch_dict["n_latent_shared"]}'
wandb_logger = WandbLogger(project=f"Dis2PVI_cE_{split_key}", name=model_name)
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
