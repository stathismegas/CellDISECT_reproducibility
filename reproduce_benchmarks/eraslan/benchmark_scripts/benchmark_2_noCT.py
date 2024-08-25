from re import split
import scanpy as sc
import pandas as pd
import numpy as np
from dis2p import dis2pvi_cE as dvi
import biolord
from scDisInFact import scdisinfact, create_scdisinfact_dataset
import gc
import torch
import random
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
import scipy

RANDOM_SEED = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata_biolord = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()
adata_biolord = adata_biolord[adata_biolord.layers['counts'].sum(1) != 0].copy()
# If sparese, convert to dense
if scipy.sparse.issparse(adata.X):
    adata.X = adata.X.todense()
    adata_biolord.X = adata_biolord.X.todense()
if scipy.sparse.issparse(adata.layers['counts']):
    adata.layers['counts'] = adata.layers['counts'].todense()
    adata_biolord.layers['counts'] = adata_biolord.layers['counts'].todense()


cats = ['tissue', 'Sample ID', 'sex', 'Age_bin']
condition_key = ['tissue', 'Sample ID', 'sex', 'Age_bin']

#scDisInfact Related Data
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
Ks = [40] + [10] * len(condition_key)
batch_size = 64
interval = 10
lr = 5e-4
batch_size = 64
adata_ = adata[adata.layers['counts'].sum(1) != 0].copy()
counts = adata_.layers['counts'].copy()
meta_cells = adata_.obs.copy()
meta_cells['one'] = pd.Categorical([1 for _ in range(adata_.n_obs)])
data_dict = create_scdisinfact_dataset(counts, meta_cells, condition_key = condition_key, batch_key = "one")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
del adata_

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
    biolord_model_path: str,
    scdisinfact_model_path: str,
):
    model = dvi.Dis2pVI_cE.load(f"{pre_path}/{dis2p_model_path}", adata=adata)
    biolord_model = biolord.Biolord.load(f"{pre_path}/{biolord_model_path}", adata=adata_biolord)
    
    scdisinfact_model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                        reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)

    scdisinfact_model.load_state_dict(torch.load(f"{pre_path}/{scdisinfact_model_path}", map_location = device))
    
    return model, biolord_model, scdisinfact_model

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

def biolord_pred(
    biolord_model,
    adata,
    cell_type_to_check,
    cov_names,
    cov_values,
    cov_values_cf,
    n_samples_from_source = None,
):
    adata.obs['idx'] = list([i for i in range(len(adata))])
    adata_ = adata[adata.obs['Broad cell type'] == cell_type_to_check].copy()
        
    source_indices = pd.DataFrame([adata_.obs[cov_name] == cov_values[i] for i, cov_name in enumerate(cov_names)]).all(0).values
    source_sub_idx = list(adata_[source_indices].obs['idx'])

    source_adata_biolord = adata_[adata_.obs['idx'].isin(source_sub_idx)]
    
    if n_samples_from_source is not None:
        random.seed(RANDOM_SEED)
        chosen_ids = random.sample(range(len(source_adata_biolord)), n_samples_from_source)
        source_adata_biolord = source_adata_biolord[chosen_ids].copy()
    biolord_preds = biolord_model.compute_prediction_adata(adata, adata_source=source_adata_biolord, target_attributes=cov_names)
    
    pred_idx = pd.DataFrame([biolord_preds.obs[cov_name] == cov_values_cf[i] for i, cov_name in enumerate(cov_names)]).all(0).values
    biolord_preds = biolord_preds[pred_idx]

    x_biolord = torch.tensor(biolord_preds.X)
    x_biolord = np.log1p(x_biolord)
                     
    return x_biolord

def scdisinfact_pred(
    scdisinfact_model,
    meta_cells,
    condition_key,
    counts,
    cell_type_to_check,
    cov_names,
    cov_values,
    cov_values_cf,
    n_samples_from_source = None,
):
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings("ignore")
    meta_cells_ = meta_cells[(meta_cells['Broad cell type'] == cell_type_to_check)]
    counts_ = counts[(meta_cells['Broad cell type'] == cell_type_to_check)]
    
    meta_cells_ = meta_cells_[condition_key + ['one']]
    input_idx = pd.DataFrame([meta_cells_[cov_name] == cov_values[i] for i, cov_name in enumerate(cov_names)]).all(0).values
    counts_input = counts_[input_idx,:]
    meta_input = meta_cells_.loc[input_idx,:]
    
    for cov in condition_key + ['one']:
        meta_input[cov] = meta_input[cov].astype(str)

    meta_output = meta_input.copy()
    for i, cov in enumerate(cov_names):
        meta_output[cov] = cov_values_cf[i]
   
    counts_predict = []
    if n_samples_from_source is not None:
        random.seed(RANDOM_SEED)
        chosen_ids = random.sample(range(len(meta_input)), n_samples_from_source)
    else:
        chosen_ids = range(len(meta_input))
    for i in tqdm(chosen_ids):
        predict_conds = [meta_output[cov][i] for cov in condition_key]
        cell_i = scdisinfact_model.predict_counts(input_counts = counts_input[i, :][None, :],
                                                  meta_cells = meta_input.iloc[[i], :], 
                                                  condition_keys = condition_key, 
                                                  batch_key = "one", 
                                                  predict_conds = predict_conds,)
        counts_predict.append(cell_i)

    counts_predict = np.concatenate(counts_predict)
    x_scdisinfact = torch.tensor(counts_predict)
    x_scdisinfact = np.log1p(x_scdisinfact)
    
    return x_scdisinfact

split_name = 'split_2'
gc.collect()

dis2p_model_path = (
    f'dis2p_cE_{split_name}/'
    f'pretrainAE_10_maxEpochs_1000_split_{split_name}_reconW_20_cfWeight_1.5_beta_0.003_clf_0.8_adv_0.015_advp_5_n_cf_1_lr_0.01_wd_0.0005_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32_n_layers_2_batch_size_256_NoCT'
)
biolord_model_path = f'biolord/eraslan_biolord_NoCT_earlierStop_basicSettings_nb_{split_name}/'
scdisinfact_model_path = f'scDisInfact/eraslan_scdisinfact_NoCT_defaultSettings_f{split_name}.pth'

model, biolord_model, scdisinfact_model = load_models(dis2p_model_path, biolord_model_path, scdisinfact_model_path)

cell_type_to_check = 'Epithelial cell (luminal)'
n_samples_from_source = min(n_samples_from_source_max, len(adata[(adata.obs['Broad cell type'] == cell_type_to_check) &
                (adata.obs['tissue'] == 'breast') & (adata.obs['sex'] == 'female')]))


cov_names = ['sex', 'tissue']
cov_values = ['female', 'breast']
cov_values_cf = ['male', 'prostate gland']

adata_ = adata[adata.obs['Broad cell type'] == cell_type_to_check].copy()
    
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
print("Getting predictions for Biolord...")
x_biolord = biolord_pred(
    biolord_model,
    adata_biolord,
    cell_type_to_check,
    cov_names=cov_names,
    cov_values=cov_values,
    cov_values_cf=cov_values_cf,
    n_samples_from_source=n_samples_from_source,
)
print("Getting predictions for scDisInfact...")
x_scdisinfact = scdisinfact_pred(
    scdisinfact_model,
    meta_cells,
    condition_key,
    counts,
    cell_type_to_check,
    cov_names,
    cov_values,
    cov_values_cf,
    n_samples_from_source=n_samples_from_source,
)

deg_list = adata.uns['rank_genes_groups_split_2']['male_prostate gland']

emd_results = {}
for n_top_deg in [20, None]:
    if n_top_deg is not None:
        degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
    else:
        degs = np.arange(adata.n_vars)
        n_top_deg = 'all'

    x_true_deg = x_true[:, degs]
    x_pred_deg = x_pred[:, degs]
    x_ctrl_deg = x_ctrl[:, degs]
    x_biolord_deg = x_biolord[:, degs]
    x_scdisinfact_deg = x_scdisinfact[:, degs]
    
    emd_results[str(n_top_deg)] = {}
    for method_name, method in zip(['Dis2P', 'Biolord', 'scdisinfact', 'Control'], [x_pred_deg, x_biolord_deg, x_scdisinfact_deg, x_ctrl_deg]):
        wd = []
        for i in range(x_true_deg.shape[1]):
            wd.append(
                wasserstein_distance(torch.tensor(x_true_deg[:, i]), torch.tensor(method[:, i]))
            )
        emd_results[str(n_top_deg)][method_name] = np.mean(wd)

emd_results = pd.DataFrame.from_dict(emd_results).T

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
    x_biolord_deg = x_biolord[:, degs]
    x_scdisinfact_deg = x_scdisinfact[:, degs]
    

    r2_mean_deg = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))
    r2_mean_base_deg = pearsonr(x_true_deg.mean(0), x_ctrl_deg.mean(0))
    r2_mean_biolord_deg = pearsonr(x_true_deg.mean(0), x_biolord_deg.mean(0))
    r2_mean_scdisinfact_deg = pearsonr(x_true_deg.mean(0), x_scdisinfact_deg.mean(0))
    
    r2_var_deg = pearsonr(x_true_deg.var(0), x_pred_deg.var(0))
    r2_var_base_deg = pearsonr(x_true_deg.var(0), x_ctrl_deg.var(0))
    r2_var_biolord_deg = pearsonr(x_true_deg.var(0), x_biolord_deg.var(0))
    r2_var_scdisinfact_deg = pearsonr(x_true_deg.var(0), x_scdisinfact_deg.var(0))
    
    r2_results[str(n_top_deg)] = {}
    r2_results[str(n_top_deg)]['Dis2P'] = r2_mean_deg[0]
    r2_results[str(n_top_deg)]['Biolord'] = r2_mean_biolord_deg[0]
    r2_results[str(n_top_deg)]['scdisinfact'] = r2_mean_scdisinfact_deg[0]
    r2_results[str(n_top_deg)]['Control'] = r2_mean_base_deg[0]

    r2_results[str(n_top_deg)]['Dis2P_var'] = r2_var_deg[0]
    r2_results[str(n_top_deg)]['Biolord_var'] = r2_var_biolord_deg[0]
    r2_results[str(n_top_deg)]['scdisinfact_var'] = r2_var_scdisinfact_deg[0]
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
    x_biolord_deg = x_biolord[:, degs]
    x_scdisinfact_deg = x_scdisinfact[:, degs]
    

    r2_mean_deg = pearsonr(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
    r2_mean_biolord_deg = pearsonr(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_biolord_deg.mean(0) - x_ctrl_deg.mean(0))
    r2_mean_scdisinfact_deg = pearsonr(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_scdisinfact_deg.mean(0) - x_ctrl_deg.mean(0))
    
    r2_var_deg = pearsonr(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))
    r2_var_biolord_deg = pearsonr(x_true_deg.var(0) - x_ctrl_deg.var(0), x_biolord_deg.var(0) - x_ctrl_deg.var(0))
    r2_var_scdisinfact_deg = pearsonr(x_true_deg.var(0) - x_ctrl_deg.var(0), x_scdisinfact_deg.var(0) - x_ctrl_deg.var(0))
    
    r2_results_subtract[str(n_top_deg)] = {}
    r2_results_subtract[str(n_top_deg)]['Dis2P'] = r2_mean_deg[0]
    r2_results_subtract[str(n_top_deg)]['Biolord'] = r2_mean_biolord_deg[0]
    r2_results_subtract[str(n_top_deg)]['scdisinfact'] = r2_mean_scdisinfact_deg[0]

    r2_results_subtract[str(n_top_deg)]['Dis2P_var'] = r2_var_deg[0]
    r2_results_subtract[str(n_top_deg)]['Biolord_var'] = r2_var_biolord_deg[0]
    r2_results_subtract[str(n_top_deg)]['scdisinfact_var'] = r2_var_scdisinfact_deg[0]
    
r2_results_subtract = pd.DataFrame.from_dict(r2_results_subtract).T

emd_results.to_csv(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/cf_results/eraslan_bingo_noCT_{split_name}_emd.csv')
r2_results.to_csv(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/cf_results/eraslan_bingo_noCT_{split_name}_pearson.csv')
r2_results_subtract.to_csv(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/cf_results/eraslan_bingo_noCT_{split_name}_delta_pearson.csv')
gc.collect()
