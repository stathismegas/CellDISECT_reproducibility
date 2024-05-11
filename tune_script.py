from dis2p.tuner_base import run_autotune
from dis2p import dis2pvi_cE as dvi

import scanpy as sc
from ray import tune

import pickle


# DATA_PATH = '/nfs/team205/sm58/packages/dis2p_trials/dis2p_reproduciblity/data/eraslan_preprocessed1200_split.h5ad'  # Change this to your desired path
DATA_PATH = '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1200_split_deg.h5ad'  # Change this to your desired path
adata = sc.read_h5ad(DATA_PATH)
# Counts should be available in the 'counts' layer
adata.X = adata.layers['counts'].copy()
sc.pp.subsample(adata, fraction=0.1)

split_key = 'split_Epithelial_luminal'

model_args = {'n_layers': tune.choice([2, 3]),
              'n_hidden': tune.choice([128]),
              'n_latent_shared': tune.choice([10, 40]),
              'n_latent_attribute': tune.sample_from(
              lambda spec: spec.config.model_args.n_latent_shared),
              'dropout_rate': tune.choice([0.2]),
              'split_key': split_key,
              'train_split': ['train'],
              'valid_split': ['val'],
              'test_split': ['test'],
              }

train_args = {
    ##################### plan_kwargs #####################
    'lr': tune.loguniform(1e-5, 1e-2),
    'weight_decay': tune.loguniform(1e-6, 1e-1),
    'new_cf_method': True,
    'lr_patience': 3,
    'lr_factor': 0.5,
    'lr_scheduler_metric': 'loss_validation',

    # 'weight_decay': tune.loguniform(1e-6, 1e-3),
    # 'n_epochs_kl_warmup': 40,
}
plan_kwargs_keys = list(train_args.keys())

trainer_actual_args = {
    # 'max_epochs': tune.choice([50, 80, 100, 150]),
    'max_epochs': 1000,
    # 'max_epochs': tune.choice([2]),
    'batch_size': tune.choice([1024]),
    # 'cf_weight': tune.uniform(1e-4, 1e0),
    'cf_weight': tune.loguniform(1e-4, 1e1),
    'beta': tune.loguniform(1e-4, 1e0),
    'clf_weight': tune.loguniform(1e-4, 1e0),
    'adv_clf_weight': tune.loguniform(1e-4, 1e0),
    'adv_period': tune.choice([1, 2, 5]),
    # 'n_cf': 1,
    'n_cf': tune.choice([1, 3]),
    'early_stopping_patience': 5,
    'early_stopping': True,
}
train_args.update(trainer_actual_args)

search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

# scheduler_kwargs = {
#    'mode': 'min',
#    'metric': 'loss_validation',
#    'max_t': 1000,
#    'grace_period': 5,
#    'reduction_factor': 3,
# }

# searcher_kwargs = {
#     'mode': 'max',
#     'metric': 'cpa_metric',
# }

# Change this to your desired categorical covariates
# cats = ['tissue', 'Sample ID', 'sex', 'Age_bin']
cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']

setup_anndata_kwargs = {
    'layer': 'counts',
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
           'rec_x_cf_validation',
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
    # scheduler="asha",
    scheduler="fifo",
    # searcher="hyperopt",
    searcher="random",
    seed=1,
    # Change this to your desired resources
    resources={"cpu": 2, "gpu": 0.2, "memory": 35 * 1024 * 1024 * 1024},
    experiment_name="dis2p_autotune",  # Change this to your desired experiment name
    logging_dir='/path/to/logs',  # Change this to your desired path
    adata_path=DATA_PATH,
    sub_sample=0.1,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=True,  # If you want to use wandb, set this to True
    wandb_name="dis2p_tune",  # Change this to your desired wandb project name
    scheduler_kwargs=scheduler_kwargs,
    plan_kwargs_keys=plan_kwargs_keys,
    # searcher_kwargs=searcher_kwargs,
)
result_grid = experiment.result_grid
with open('result_grid.pkl', 'wb') as f:
    pickle.dump(result_grid, f)
