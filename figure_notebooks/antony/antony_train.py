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
print(torch.cuda.is_available())

root_dir = '/lustre/scratch126/cellgen/team298/ar32/farm_job_objects_and_outputs_temp/new_dis2p_20240619/prepare_data/'
data = 'dis2p_prepped_20240619'

adata = sc.read(root_dir + data + '.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

# specify name of dataset 
data_name = 'IA'

# specify attributes
cats = ['integration_donor', 'integration_biological_unit', 'integration_sample_status', 'integration_library_platform_coarse',
        'organ','bin_age']


arch_dict = {'n_layers': 6,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 1000,
 'batch_size': 1024,
 'recon_weight': 28.55,
 'cf_weight': 0.7873,
 'beta': 0.003,
 'clf_weight': 0.8149,
 'adv_clf_weight': 0.0138,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 15,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
}

plan_kwargs = {
 'lr': 0.001,
 'weight_decay': 0.0000778,
 'new_cf_method': True,
 'lr_patience': 6,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


module_name = f'Antony'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)


# specify a name for your model
model_name =  f'maxEpoch_{train_dict["max_epochs"]}_reconW_{train_dict["recon_weight"]}_cfWeight_{train_dict["cf_weight"]}_beta_{train_dict["beta"]}_clf_{train_dict["clf_weight"]}_adv_{train_dict["adv_clf_weight"]}_advp_{train_dict["adv_period"]}_n_cf_{train_dict["n_cf"]}_lr_{plan_kwargs["lr"]}_wd_{plan_kwargs["weight_decay"]}_new_cf_{plan_kwargs["new_cf_method"]}_dropout_{arch_dict["dropout_rate"]}_n_hidden_{arch_dict["n_hidden"]}_n_latent_{arch_dict["n_latent_shared"]}_n_layers_{arch_dict["n_layers"]}'
wandb_logger = WandbLogger(project=f"Antony_Dis2PVI_cE", name=model_name)
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
                       **arch_dict)
model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)
