from celldisect.tuner_base import run_autotune
from celldisect import CellDISECT

import scanpy as sc
from ray import tune

import pickle

from metric_utils import asw_report, knn_purity


DATA_PATH = '/lustre/scratch126/cellgen/team298/ar32/farm_job_objects_and_outputs_temp/dis2p_20240703_suo/prepare_data/dis2p_suo_prepped_5_percent.h5ad'  # Change this to your desired path
adata = sc.read_h5ad(DATA_PATH)
# Counts should be available in the 'counts' layer
# adata.X = adata.layers['counts'].copy()

# split_key = 'split_CD4 T'

model_args = {'n_layers': 2,
              'n_hidden': 128,
              'n_latent_shared': 32,
              'n_latent_attribute': tune.sample_from(
              lambda spec: spec.config.model_args.n_latent_shared),
              'dropout_rate': 0.2,
            #   'split_key': split_key,
            #   'train_split': ['train'],
            #   'valid_split': ['valid'],
            #   'test_split': ['ood'],
              }

train_args = {
    ##################### plan_kwargs #####################
    'lr': tune.loguniform(1e-3, 1e-1),
    'weight_decay': tune.loguniform(1e-6, 1e-1),
    'new_cf_method': True,
    'lr_patience': 6,
    'lr_factor': 0.5,
    'lr_scheduler_metric': 'loss_validation',
    'n_epochs_kl_warmup': 10,
}
plan_kwargs_keys = list(train_args.keys())

trainer_actual_args = {
    'max_epochs': 350,
    'batch_size': 256,
    'recon_weight': tune.loguniform(5, 3e1),
    'cf_weight': tune.loguniform(5e-1, 1e1),
    'beta': 0.003,
    'clf_weight': 0,
    'adv_clf_weight': 0.014,
    'adv_period': 5,
    'n_cf': 1,
    # 'n_cf': tune.choice([1, 3]),
    'early_stopping_patience': 15,
    'early_stopping': True,
    'save_best': True,
    'kappa_optimizer2': False,
    'n_epochs_pretrain_ae': tune.choice([0, 10]),
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
cats = ['integration_donor', 'integration_library_platform_coarse',
        'organ', 'bin_age']

setup_anndata_kwargs = {
    'layer': 'counts',
    'categorical_covariate_keys': cats,
    'continuous_covariate_keys': []
}
model = CellDISECT
model.setup_anndata(adata, **setup_anndata_kwargs)

x_loss = [f'x_{i}_validation' for i in range(len(cats)+1)]
z_loss = [f'z_{i}_validation' for i in range(1, len(cats)+1)]
metrics = ['loss_validation', # The first one is the one that will be used for optimization "MAIN ONE"
           'acc_validation',
           'f1_validation',
           'adv_ce_validation',
           'adv_acc_validation',
           'adv_f1_validation',
           'rec_x_cf_validation',
           'loss_train']
metrics += x_loss
metrics += z_loss

def evaluation_function(model, adata):
    report_dict = {}
    
    ## ASW
    asw_results = asw_report(model, adata, cats, cats_to_check=['organ'], label_key='anno_lvl_2_final_clean_orig_author')
    report_dict.update(asw_results)
    
    ## Knn purity
    kpurity = knn_purity(model, adata, 'organ', cats, n_neighbors=30)
    report_dict['KNN_Purity'] = kpurity
    return report_dict


experiment = run_autotune(
    model_cls=model,
    data=adata,
    metrics=metrics,
    mode="min",
    search_space=search_space,
    # Change this to your desired number of samples (Number of runs)
    num_samples=400,
    # scheduler="asha",
    scheduler="fifo",
    # searcher="hyperopt",
    searcher="random",
    seed=1,
    # Change this to your desired resources
    resources={"cpu": 1, "gpu": 0.1, "memory": 64 * 1024 * 1024 * 1024},
    experiment_name="suo_hyperTune",  # Change this to your desired experiment name
    logging_dir='/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/antony/hptune/',  # Change this to your desired path
    adata_path=DATA_PATH,
    sub_sample=None,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=True,  # If you want to use wandb, set this to True
    wandb_name="suo_hyperTune",  # Change this to your desired wandb project name
    # scheduler_kwargs=scheduler_kwargs,
    plan_kwargs_keys=plan_kwargs_keys,
    # searcher_kwargs=searcher_kwargs,
    evaluation_func=evaluation_function,
)
result_grid = experiment.result_grid
with open('result_grid.pkl', 'wb') as f:
    pickle.dump(result_grid, f)
