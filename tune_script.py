from dis2p_defunct.tuner_base import run_autotune
from dis2p_defunct import dis2pvi_cE as dvi

import scanpy as sc
from ray import tune
import numpy as np

import pickle


DATA_PATH = '/PATH/TO/DATA.h5ad'  # Change this to your desired path
adata = sc.read_h5ad(DATA_PATH)
# Counts should be available in the 'counts' layer
adata.X = adata.layers['counts'].copy()
sc.pp.subsample(adata, fraction=0.1)

model_args = {'n_layers': tune.choice([1, 2, 3]),
              'n_hidden': tune.choice([128, 256, 512, 1024]),
              'n_latent_shared': tune.choice([10, 20, 30, 40, 50]),
              'n_latent_attribute': tune.sample_from(
              lambda spec: spec.config.model_args.n_latent_shared),
              'dropout_rate': tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
              }

train_args = {
    ##################### plan_kwargs #####################
    'lr': tune.loguniform(1e-5, 1e-2),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
}
plan_kwargs_keys = list(train_args.keys())

trainer_actual_args = {
    'max_epochs': tune.choice([50, 80, 100, 150]),
    'batch_size': tune.choice([512]),
    'cf_weight': tune.loguniform(1, 1e2),
    'beta': tune.loguniform(1e-2, 1),
    'clf_weight': tune.loguniform(1e-1, 1e2),
    'adv_clf_weight': tune.loguniform(1e-1, 1e2),
    'adv_period': tune.choice([1, 2, 3]),
    'n_cf': tune.choice([1, 2, 3]),
}
train_args.update(trainer_actual_args)

search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

scheduler_kwargs = {
    # 'mode': 'max',
    # 'metric': 'cpa_metric',
    'max_t': 1000,
    'grace_period': 5,
    'reduction_factor': 4,
}

# searcher_kwargs = {
#     'mode': 'max',
#     'metric': 'cpa_metric',
# }

# Change this to your desired categorical covariates
cats = ['tissue', 'Sample ID', 'cell_type', 'sex', 'Age_bin']

setup_anndata_kwargs = {
    'categorical_covariate_keys': cats,
    'continuous_covariate_keys': []
}
model = dvi.Dis2pVI_cE
model.setup_anndata(adata, **setup_anndata_kwargs)

x_loss = [f'x_{i}_validation' for i in range(len(cats)+1)]
z_loss = [f'z_{i}_validation' for i in range(1, len(cats)+1)]
metrics = ['loss_validation', # The first one (cpa_metric) is the one that will be used for optimization "MAIN ONE"
           'acc_validation',
           'f1_validation',
           'adv_ce_validation',
           'adv_acc_validation',
           'adv_f1_validation',
           'loss_train']
metrics += x_loss
metrics += z_loss

experiment = run_autotune(
    model_cls=model,
    data=adata,
    metrics=metrics,
    mode="min",
    search_space=search_space,
    # Change this to your desired number of samples (Number of runs)
    num_samples=5000,
    scheduler="asha",
    searcher="hyperopt",
    seed=1,
    # Change this to your desired resources
    resources={"cpu": 40, "gpu": 0.2, "memory": 16000},
    experiment_name="dis2p_autotune",  # Change this to your desired experiment name
    logging_dir='/PATH/TO/LOGS/',  # Change this to your desired path
    adata_path=DATA_PATH,
    sub_sample=0.1,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=False,  # If you want to use wandb, set this to True
    wandb_name="dis2p_tune",  # Change this to your desired wandb project name
    scheduler_kwargs=scheduler_kwargs,
    plan_kwargs_keys=plan_kwargs_keys,
    # searcher_kwargs=searcher_kwargs,
)
result_grid = experiment.result_grid
with open('result_grid.pkl', 'wb') as f:
    pickle.dump(result_grid, f)
