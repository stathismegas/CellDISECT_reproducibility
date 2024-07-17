import sys
recon_weight = float(sys.argv[1])
cf_weight = float(sys.argv[2])

from matplotlib.pylab import f
import scanpy as sc
import pandas as pd
import numpy as np
from dis2p import dis2pvi_cE as dvi
import gc
from scipy.stats import pearsonr
import scipy

RANDOM_SEED = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()
if scipy.sparse.issparse(adata.X):
    adata.X = adata.X.todense()
if scipy.sparse.issparse(adata.layers['counts']):
    adata.layers['counts'] = adata.layers['counts'].todense()

cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']
gc.collect()
from typing import NamedTuple

class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    LATENT_MODE_KEY: str = "latent_mode"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()


n_samples_from_source_max = 500

pre_path = '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/'

def load_models(
    dis2p_model_path: str,
):
    model = dvi.Dis2pVI_cE.load(f"{pre_path}/{dis2p_model_path}", adata=adata)

    return model

def dis2p_pred(
    model,
    adata,
    cell_type_to_check,
    cov_names,
    cov_values: str,
    cov_values_cf: str,
    cats: list[str],
    n_samples_from_source = None,
):
    x_ctrl, x_true, x_pred = model.predict_counterfactuals(
                                            adata,
                                            cov_names=cov_names,
                                            cov_values=cov_values,
                                            cov_values_cf=cov_values_cf,
                                            cats=cats,
                                            n_samples_from_source=n_samples_from_source,
                                            seed=RANDOM_SEED,
                                            )

    x_ctrl, x_true, x_pred = np.log1p(x_ctrl), np.log1p(x_true), np.log1p(x_pred)
    return x_ctrl, x_true, x_pred


split_name = 'split_1'
gc.collect()


arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 100,
 'batch_size': 256,
 'recon_weight': recon_weight,
 'cf_weight': cf_weight,
 'beta': 0.003,
 'clf_weight': 0.8,
 'adv_clf_weight': 0.015,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 15,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
 'n_epochs_pretrain_ae': 10,
}

plan_kwargs = {
 'lr': 0.01,
 'weight_decay': 0.0005,
 'new_cf_method': True,
 'lr_patience': 6,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}


# specify a name for your model
model_name = (
    f'pretrainAE_{train_dict["n_epochs_pretrain_ae"]}_'
    f'maxEpochs_{train_dict["max_epochs"]}_split_{split_name}_'
    f'reconW_{train_dict["recon_weight"]}_'
    f'cfWeight_{train_dict["cf_weight"]}_'
    f'beta_{train_dict["beta"]}_'
    f'clf_{train_dict["clf_weight"]}_'
    f'adv_{train_dict["adv_clf_weight"]}_'
    f'advp_{train_dict["adv_period"]}_'
    f'n_cf_{train_dict["n_cf"]}_'
    f'lr_{plan_kwargs["lr"]}_'
    f'wd_{plan_kwargs["weight_decay"]}_'
    f'new_cf_{plan_kwargs["new_cf_method"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}'
)

module_name = f'dis2p_cE_{split_name}_cf_generalization'
dis2p_model_path = f'{module_name}/{model_name}'
model = load_models(dis2p_model_path)

cell_type_to_check = 'Immune (DC/macrophage)'

cov_names = ['sex']
cov_values = ['female']
cov_values_cf = ['male']

adata_ = adata[(adata.obs['Broad cell type'] == cell_type_to_check) &
                (adata.obs['tissue'] == 'lingula of left lung')].copy()

n_samples_from_source = min(n_samples_from_source_max, len(adata[(adata.obs['Broad cell type'] == cell_type_to_check) &
                (adata.obs['tissue'] == 'lingula of left lung') & (adata.obs['sex'] == 'female')]))

print(f"Predicting for {split_name}")
print("Getting predictions for Dis2P...")
x_ctrl, x_true, x_pred = dis2p_pred(
    model,
    adata_,
    cell_type_to_check,
    cov_names,
    cov_values,
    cov_values_cf,
    cats,
    n_samples_from_source
    )

deg_list = adata.uns['rank_genes_groups_split_1']['Immune (DC']['macrophage)_male']
r2_results = {}
for n_top_deg in [20, None]:
    if n_top_deg is not None:
        degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
    else:
        degs = np.arange(adata.n_vars)
        n_top_deg = 'all'

    x_true_deg = x_true[:, degs]
    x_pred_deg = x_pred[:, degs]
    x_ctrl_deg = x_ctrl[:, degs]


    r2_mean_deg = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))
    r2_mean_base_deg = pearsonr(x_true_deg.mean(0), x_ctrl_deg.mean(0))

    r2_var_deg = pearsonr(x_true_deg.var(0), x_pred_deg.var(0))
    r2_var_base_deg = pearsonr(x_true_deg.var(0), x_ctrl_deg.var(0))

    r2_results[str(n_top_deg)] = {}
    r2_results[str(n_top_deg)]['Dis2P'] = r2_mean_deg[0]
    r2_results[str(n_top_deg)]['Control'] = r2_mean_base_deg[0]

    r2_results[str(n_top_deg)]['Dis2P_var'] = r2_var_deg[0]
    r2_results[str(n_top_deg)]['Control_var'] = r2_var_base_deg[0]

r2_results = pd.DataFrame.from_dict(r2_results).T

#######################################################################
r2_results_subtract = {}
for n_top_deg in [20, None]:
    if n_top_deg is not None:
        degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
    else:
        degs = np.arange(adata.n_vars)
        n_top_deg = 'all'

    x_true_deg = x_true[:, degs]
    x_pred_deg = x_pred[:, degs]
    x_ctrl_deg = x_ctrl[:, degs]


    r2_mean_deg = pearsonr(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))

    r2_var_deg = pearsonr(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

    r2_results_subtract[str(n_top_deg)] = {}
    r2_results_subtract[str(n_top_deg)]['Dis2P'] = r2_mean_deg[0]

    r2_results_subtract[str(n_top_deg)]['Dis2P_var'] = r2_var_deg[0]

r2_results_subtract = pd.DataFrame.from_dict(r2_results_subtract).T

experiment = f'bingo_{recon_weight}_{cf_weight}'
r2_results.to_csv(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/ablation_cf_results/eraslan_dis2p_{experiment}_{split_name}_pearson.csv')
r2_results_subtract.to_csv(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/ablation_cf_results/eraslan_dis2p_{experiment}_{split_name}_delta_pearson.csv')
gc.collect()
                 