from celldisect.tuner_base import run_autotune
from celldisect import CellDISECT

import scanpy as sc
from ray import tune

import pickle

from metric_utils import pred_our_ood_avg, counterfactual_report, asw_report


DATA_PATH = '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad'  # Change this to your desired path
adata = sc.read_h5ad(DATA_PATH)
# Counts should be available in the 'counts' layer
adata.X = adata.layers['counts'].copy()

split_key = 'split_CD4 T'

model_args = {'n_layers': tune.choice([2, 4]),
              'n_hidden': tune.choice([128]),
              'n_latent_shared': tune.choice([32, 64]),
              'n_latent_attribute': tune.sample_from(
              lambda spec: spec.config.model_args.n_latent_shared),
              'dropout_rate': 0.2,
              'split_key': split_key,
              'train_split': ['train'],
              'valid_split': ['valid'],
              'test_split': ['ood'],
              }

train_args = {
    ##################### plan_kwargs #####################
    'lr': tune.loguniform(1e-5, 1e-2),
    'weight_decay': tune.loguniform(1e-6, 1e-1),
    'new_cf_method': True,
    'lr_patience': 6,
    'lr_factor': 0.5,
    'lr_scheduler_metric': 'loss_validation',
    'n_epochs_kl_warmup': 10,
}
plan_kwargs_keys = list(train_args.keys())

trainer_actual_args = {
    # 'max_epochs': tune.choice([50, 80, 100, 150]),
    'max_epochs': 350,
    # 'max_epochs': tune.choice([2]),
    'batch_size': 128,
    'recon_weight': tune.loguniform(5, 2e1),
    'cf_weight': tune.loguniform(5e-1, 1e1),
    'beta': 0.003,
    # 'beta': tune.loguniform(1e-4, 1e0),
    'clf_weight': tune.loguniform(1e-1, 1e1),
    'adv_clf_weight': tune.loguniform(1e-2, 1),
    'adv_period': 5,
    'n_cf': 1,
    # 'n_cf': tune.choice([1, 3]),
    'early_stopping_patience': 15,
    'early_stopping': True,
    'save_best': True,
    'kappa_optimizer2': False,
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
cats = ['cell_type', 'condition']

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
    
    cell_type_to_check = ['CD4 T', 'CD14 Mono'] # First one is ood second one is in-distribution
    key_added = 'rank_genes_groups_pval0.05'

    cov_names = ['condition']
    cov_values = ['ctrl']
    cov_values_cf = ['stimulated']
    n_samples_from_source = None
    
    ## OOD
    x_ctrl, x_true, x_pred, _, _ = pred_our_ood_avg(model,
                                              adata[adata.obs['cell_type'] == cell_type_to_check[0]].copy(),
                                              cov_names=cov_names,
                                              cov_values=cov_values,
                                              cov_values_cf=cov_values_cf,
                                              cats=cats,
                                              n_samples_from_source=n_samples_from_source,
                                              dec_b=0,
                                              dec_e=None,
                                             )
    group_key = f'{cell_type_to_check[0]}_stimulated'
    r2_results, r2_results_subtract = counterfactual_report(adata,
                                                            key_added, 
                                                            x_true,
                                                            x_pred,
                                                            x_ctrl,
                                                            group_key, 
                                                            name_to_add='OOD')
    r2_results = r2_results.to_dict()
    r2_results_subtract = r2_results_subtract.to_dict()
    report_dict.update(r2_results)
    report_dict.update(r2_results_subtract)
    
    ## In-distribution
    x_ctrl, x_true, x_pred, _, _ = pred_our_ood_avg(model,
                                              adata[(adata.obs['cell_type'] == cell_type_to_check[1]) &
                                                    (adata.obs[split_key] == 'valid')].copy(),
                                              cov_names=cov_names,
                                              cov_values=cov_values,
                                              cov_values_cf=cov_values_cf,
                                              cats=cats,
                                              n_samples_from_source=n_samples_from_source,
                                              dec_b=0,
                                              dec_e=None,
                                             )
    group_key = f'{cell_type_to_check[1]}_stimulated'
    r2_results, r2_results_subtract = counterfactual_report(adata,
                                                            key_added, 
                                                            x_true,
                                                            x_pred,
                                                            x_ctrl,
                                                            group_key, 
                                                            name_to_add='Validation')
    r2_results = r2_results.to_dict()
    r2_results_subtract = r2_results_subtract.to_dict()
    report_dict.update(r2_results)
    report_dict.update(r2_results_subtract)
    
    
    ## ASW
    asw_results = asw_report(model, adata, cats)
    report_dict.update(asw_results)
    
    return report_dict
        

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
    resources={"cpu": 1, "gpu": 0.1, "memory": 64 * 1024 * 1024 * 1024},
    experiment_name="kang_hyperTune",  # Change this to your desired experiment name
    logging_dir='/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/CellDISECT_reproducibility/figure_notebooks/kang/kang_split_CD4_T',  # Change this to your desired path
    adata_path=DATA_PATH,
    sub_sample=None,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=True,  # If you want to use wandb, set this to True
    wandb_name="kang_hyperTune",  # Change this to your desired wandb project name
    # scheduler_kwargs=scheduler_kwargs,
    plan_kwargs_keys=plan_kwargs_keys,
    # searcher_kwargs=searcher_kwargs,
    evaluation_func=evaluation_function,
)
result_grid = experiment.result_grid
with open('result_grid.pkl', 'wb') as f:
    pickle.dump(result_grid, f)
