import torch
import numpy as np
import pandas as pd
import random
import anndata as ad
from scipy.stats import pearsonr
import scib


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


def pred_our_ood_avg(model,
                adata,  # OOD Adata
                cov_names,
                cov_values: str,
                cov_values_cf: str,
                cats: list[str],
                n_samples_from_source = None,
                n_samples: int = 1000,
                dec_b=0,
                dec_e=None):
    adata.X = adata.layers['counts'].copy()
    adata.obs['idx'] = [i for i in range(len(adata))]

    true_indices = pd.DataFrame([adata.obs[cov_name] == cov_values_cf[i] for i, cov_name in enumerate(cov_names)]).all(0).values
    true_idx = list(adata[true_indices].obs['idx'])
    
    source_indices = pd.DataFrame([adata.obs[cov_name] == cov_values[i] for i, cov_name in enumerate(cov_names)]).all(0).values
    source_idx = list(adata[source_indices].obs['idx'])

    true_adata = adata[adata.obs['idx'].isin(true_idx)].copy()
    source_adata = adata[adata.obs['idx'].isin(source_idx)].copy()

    if n_samples_from_source is not None:
        random.seed(0)
        chosen_ids = random.sample(range(len(source_adata)), n_samples_from_source)
        source_adata = source_adata[chosen_ids].copy()
    adata_cf = source_adata.copy()
        
    
    for i, cov_name in enumerate(cov_names):
        adata_cf.obs.loc[:, cov_name] = pd.Categorical(
            [cov_values_cf[i] for _ in adata_cf.obs[cov_name]])
    batch_size = len(adata_cf)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.setup_anndata(
        adata_cf,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    adata_cf = model._validate_anndata(adata_cf)
    source_adata = model._validate_anndata(source_adata)
    # print("Data loader OOD")
    scdl_cf = model._make_data_loader(
        adata=adata_cf, batch_size=batch_size
    )
    scdl = model._make_data_loader(
        adata=source_adata, batch_size=batch_size
    )
    # cov_idx = cats.index(cov_name)
    px_cf_mean_list = []
    for tensors, tensors_cf in zip(scdl, scdl_cf):
        x, pxs_cf = model.module.sub_forward_cf_avg(
                            x=tensors[REGISTRY_KEYS.X_KEY].to(device),
                            cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY].to(device),
                            cat_covs_cf=tensors_cf[REGISTRY_KEYS.CAT_COVS_KEY].to(device))

        # for px_cf in pxs_cf:
        #     samples = []
        #     if px_cf is None:
        #         continue
        #     for _ in range(n_samples):
        #         samples.append(px_cf.sample().to('cpu'))
        #     samples = torch.stack(samples, dim=0)
        #     x_cf = torch.mean(samples, dim=0)
        #     print(samples.shape)
        #     print(x_cf.shape)
        #     px_cf_mean_list.append(x_cf)
        for px_cf in pxs_cf[dec_b: dec_e]:
            if px_cf is None:
                continue
            x_cf = px_cf.mu
            px_cf_mean_list.append(x_cf)
    if len(px_cf_mean_list) > 1:
        px_cf_mean_tensor = torch.stack(px_cf_mean_list, dim=0)
        px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)
    else:
        px_cf_mean_tensor = px_cf_mean_list[0][None, :, :]
        px_cf_mean_pred = px_cf_mean_list[0]
    

    px_cf_mean_pred = px_cf_mean_pred.to('cpu').detach().numpy()
    px_cf_mean_tensor = px_cf_mean_tensor.to('cpu').detach().numpy()

    px_cf_mean_tensor = ad.AnnData(px_cf_mean_pred)
    px_cf_mean_tensor = torch.tensor(px_cf_mean_tensor.X)

    true_x_count = torch.tensor(true_adata.X)
    cf_x_count = torch.tensor(source_adata.X)

    x_true = true_x_count
    x_pred = px_cf_mean_tensor
    x_ctrl = cf_x_count

    x_true = np.log1p(x_true)
    x_pred = np.log1p(x_pred)
    x_ctrl = np.log1p(x_ctrl)
    return x_ctrl, x_true, x_pred, px_cf_mean_tensor, true_x_count


def counterfactual_report(adata, key_added, x_true, x_pred, x_ctrl, group_key, name_to_add=None):
    if name_to_add is None:
        name_to_add = group_key
    deg_list = adata.uns[key_added][group_key]
    r2_results = {}
    for n_top_deg in [20, None]:
        if n_top_deg is not None:
            degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            degs = np.arange(adata.n_vars)
            n_top_deg = 'all'

        x_true_deg = x_true[:, degs]
        x_pred_deg = x_pred[:, degs]

        r2_mean_deg = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))
        
        r2_var_deg = pearsonr(x_true_deg.var(0), x_pred_deg.var(0))
        
        r2_results[str(n_top_deg)] = {}
        r2_results[str(n_top_deg)][f'Prediction_{name_to_add}'] = r2_mean_deg[0]

        r2_results[str(n_top_deg)][f'Prediction_var_{name_to_add}'] = r2_var_deg[0]
        
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
        r2_results_subtract[str(n_top_deg)][f'DeltaPrediction_{name_to_add}'] = r2_mean_deg[0]

        r2_results_subtract[str(n_top_deg)][f'DeltaPrediction_var_{name_to_add}'] = r2_var_deg[0]
        
    r2_results_subtract = pd.DataFrame.from_dict(r2_results_subtract).T
    
    return r2_results, r2_results_subtract
    
    
def asw_report(model, adata, cats):
    asw_results = {}
    bio_results = []
    batch_resutls = []
    # Z_0
    adata.obsm[f'dis2p_cE_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

    for i in range(len(cats)):
        null_idx = [s for s in range(len(cats)) if s != i]
        # Z_i
        adata.obsm[f'dis2p_cE_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)

    for i in range(len(cats)):
        label_key = cats[i]
        bio = scib.metrics.silhouette(adata, label_key, f'dis2p_cE_Z_{i+1}', metric='euclidean', scale=True)
        bio_results.append(bio)
        for j in range(len(cats)):
            if j == i:
                continue
            label_key = cats[j]
            batch = scib.metrics.silhouette(adata, label_key, f'dis2p_cE_Z_{i+1}', metric='euclidean', scale=True)
            batch_resutls.append(batch)
            
    asw_results['ASW_bio'] = np.mean(bio_results)
    asw_results['ASW_batch'] = np.mean(batch_resutls)
    return asw_results
