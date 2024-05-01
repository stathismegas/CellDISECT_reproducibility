from typing import List, Tuple
import numpy as np
import torch
from sklearn.metrics import r2_score
from anndata import AnnData
import scanpy as sc
import itertools
import pandas as pd
import anndata as ad
from scvi import REGISTRY_KEYS
from scipy.sparse import issparse
from scipy.stats import wasserstein_distance
import scib
from scipy.stats import pearsonr
# from scfair.fairvi import FairVI

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


# def latent(
#         adata: AnnData,
#         cats: List[str],
#         new_model_name='',
#         pre_path: str = '.',
#         idx_cf_tensor_path: str = '',
#         plot_umap: bool = True,
#         **train_dict,
# ):
#     try:
#         model = FairVI.load(f"{pre_path}/{new_model_name}", adata=adata)

#     except:

#         FairVI.setup_anndata(
#             adata,
#             layer='counts',
#             categorical_covariate_keys=cats,
#             continuous_covariate_keys=[]
#         )
#         model = FairVI(adata, idx_cf_tensor_path=idx_cf_tensor_path)
#         model.train(**train_dict)
#         model.save(f"{pre_path}/{new_model_name}")

#     model.idx_cf_tensor_path = idx_cf_tensor_path

#     print(f"---latent 0 computation started---")
#     latent = model.get_latent_representation(
#         nullify_cat_covs_indices=list(range(len(cats))))
#     adata.obsm[f"Z_0"] = latent

#     for c in range(len(cats)):
#         print(f"---latent {c + 1} computation started---")
#         null_idx = [s for s in range(len(cats)) if s != c]
#         latent = model.get_latent_representation(
#             nullify_cat_covs_indices=null_idx, nullify_shared=True)
#         adata.obsm[f"Z_{c + 1}"] = latent

#         # concat all Z except c
#         null_idx = [c]
#         latent = model.get_latent_representation(
#             nullify_cat_covs_indices=null_idx, nullify_shared=False)
#         adata.obsm[f"Z_not_{c + 1}"] = latent

#     print(f"---latent computation completed---")

#     if plot_umap:

#         print("---UMAP computation started---")

#         for i in range(len(cats) + 1):  # loop over Z_shared and all Z_i

#             latent_name = f'Z_{i}'

#             print(f"---UMAP for {latent_name}---")

#             sc.pp.neighbors(adata, use_rep=f"{latent_name}")
#             sc.tl.umap(adata)

#             sc.pl.umap(
#                 adata,
#                 color=cats,
#                 ncols=len(cats),
#                 frameon=False,
#             )

#         print("---UMAP computation completed---")

#     return model, adata


# def scib_benchmark(Benchmarker, adata, cats, batch_key_idx, label_key_idx):
#     batch_key = cats[batch_key_idx]
#     label_key = cats[label_key_idx]

#     print(
#         f'scib benchmark metrics for (batch_key={batch_key}, label_key={label_key})')

#     bm = Benchmarker(
#         adata,
#         batch_key=batch_key,
#         label_key=label_key,
#         embedding_obsm_keys=[f"Z_{c}" for c in range(len(cats) + 1)],
#         n_jobs=-1,
#     )
#     bm.benchmark()

#     bm.plot_results_table(min_max_scale=False)


# def ood_for_given_covs_1(
#         adata: AnnData,
#         cats: List[str],
#         new_model_name='',
#         pre_path: str = '.',
#         idx_cf_tensor_path: str = '',
#         cov_idx: int = 0,
#         cov_value_idx: int = 1,
#         cov_value_cf_idx: int = 0,
#         other_covs_values: Tuple = (0,),
#         n_top_deg: int = 100,
#         **train_dict,
# ):
#     # remove cells with other_covs_values and cov_value_cf

#     cov_name = cats[cov_idx]
#     cov_value = tuple(set(adata.obs[cats[cov_idx]]))[cov_value_idx]
#     cov_value_cf = tuple(set(adata.obs[cats[cov_idx]]))[cov_value_cf_idx]

#     other_cats = [c for c in cats if c != cov_name]

#     train_sub_idx = []
#     true_idx = []
#     source_sub_idx = []
#     cell_to_idx = torch.tensor(list(range(adata.n_obs))).to(device)
#     cell_to_source_sub_idx = torch.tensor(list(range(adata.n_obs))).to(device)
#     for i in range(adata.n_obs):
#         cov_value_i = adata[i].obs[cats[cov_idx]][0]

#         if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
#             if cov_value_i == cov_value_cf:
#                 true_idx.append(i)
#             else:
#                 train_sub_idx.append(i)
#                 if cov_value_i == cov_value:
#                     source_sub_idx.append(i)
#         else:
#             train_sub_idx.append(i)

#         cell_to_idx[i] = len(train_sub_idx) - 1
#         cell_to_source_sub_idx[i] = len(source_sub_idx) - 1

#     idx_to_cell = torch.tensor(train_sub_idx).to(device)

#     adata_sub = adata[train_sub_idx]

#     cov_str_cf = cats[cov_idx] + ' = ' + \
#         str(cov_value) + ' to ' + str(cov_value_cf)
#     other_covs_str = ', '.join(
#         c + ' = ' + str(other_covs_values[k]) for k, c in enumerate(other_cats))

#     app_str = f' cf ' + cov_str_cf + ', ' + other_covs_str
#     sub_model_name = new_model_name + app_str
#     sub_idx_cf_tensor_path = idx_cf_tensor_path + app_str + '.pt'

#     try:
#         sub_idx_cf_tensor = torch.load(sub_idx_cf_tensor_path)
#     except:
#         path = idx_cf_tensor_path
#         if not path.endswith('.pt'):
#             path += '.pt'
#         sub_idx_cf_tensor = torch.load(path)
#         dim = sub_idx_cf_tensor.size()

#         for i in range(dim[0]):
#             for j in range(dim[1]):
#                 cf_idx = int(sub_idx_cf_tensor[i][j])
#                 if all(adata[cf_idx].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)) and \
#                         (adata[cf_idx].obs[cats[cov_idx]][0] == cov_value_cf):
#                     sub_idx_cf_tensor[i][j] = i

#         torch.save(sub_idx_cf_tensor, sub_idx_cf_tensor_path)

#     try:
#         model = FairVI.load(f"{pre_path}/{sub_model_name}", adata=adata_sub)
#     except:
#         adata_sub = adata_sub.copy()
#         FairVI.setup_anndata(
#             adata_sub,
#             layer='counts',
#             categorical_covariate_keys=cats,
#             continuous_covariate_keys=[]
#         )
#         model = FairVI(adata_sub, idx_cf_tensor_path=sub_idx_cf_tensor_path,
#                        cell_to_idx=cell_to_idx, idx_to_cell=idx_to_cell)
#         model.train(**train_dict)
#         try:
#             model.save(f"{pre_path}/{sub_model_name}")
#         except:
#             pass

#     source_adata = adata[source_sub_idx]

#     model.idx_cf_tensor_path = sub_idx_cf_tensor_path
#     model.cell_to_idx = cell_to_source_sub_idx
#     model.idx_to_cell = torch.tensor(source_sub_idx).to(device)

#     px_cf_mean_pred = model.predict_given_covs(adata=source_adata, cats=cats, cov_idx=cov_idx,
#                                                cov_value_cf=cov_value_cf).to('cpu')

#     true_x_count = torch.tensor(adata.layers["counts"][true_idx].toarray())
#     true_x_counts_mean = torch.mean(true_x_count, dim=0).to('cpu')

#     print(f'Counterfactual prediction for {cov_str_cf}, and {other_covs_str}')

#     r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean,
#             px_cf_mean_pred, n_top_deg=n_top_deg)

#     return true_x_counts_mean, px_cf_mean_pred


# def ood_for_given_covs_2(
#         adata: AnnData,
#         cats: List[str],
#         new_model_name='',
#         pre_path: str = '.',
#         idx_cf_tensor_path: str = '',
#         cov_idx: int = 0,
#         cov_value: str = '',
#         cov_value_cf: str = '',
#         other_covs_values: Tuple = (0,),
#         n_top_deg: int = 100,
#         **train_dict,
# ):
#     # remove all cells with other_covs_values

#     cov_name = cats[cov_idx]

#     other_cats = [c for c in cats if c != cov_name]

#     train_sub_idx = []
#     true_idx = []
#     source_sub_idx = []
#     cell_to_idx = torch.tensor(list(range(adata.n_obs))).to(device)
#     cell_to_source_sub_idx = torch.tensor(list(range(adata.n_obs))).to(device)

#     for i in range(adata.n_obs):
#         cov_value_i = adata[i].obs[cats[cov_idx]][0]

#         if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
#             if cov_value_i == cov_value_cf:
#                 true_idx.append(i)
#             elif cov_value_i == cov_value:
#                 source_sub_idx.append(i)
#         else:
#             train_sub_idx.append(i)

#         cell_to_idx[i] = len(train_sub_idx) - 1
#         cell_to_source_sub_idx[i] = len(source_sub_idx) - 1

#     idx_to_cell = torch.tensor(train_sub_idx).to(device)
#     adata_sub = adata[train_sub_idx]

#     cov_str_cf = cats[cov_idx] + ' = ' + \
#         str(cov_value) + ' to ' + str(cov_value_cf)
#     other_covs_str = ', '.join(
#         c + ' = ' + str(other_covs_values[k]) for k, c in enumerate(other_cats))

#     app_str = f'--cf--' + cov_str_cf + ', ' + other_covs_str
#     sub_model_name = new_model_name + app_str
#     sub_idx_cf_tensor_path = idx_cf_tensor_path + app_str + '.pt'

#     try:
#         sub_idx_cf_tensor = torch.load(sub_idx_cf_tensor_path)
#     except:
#         path = idx_cf_tensor_path
#         if not path.endswith('.pt'):
#             path += '.pt'
#         sub_idx_cf_tensor = torch.load(path)
#         dim = sub_idx_cf_tensor.size()

#         for i in range(dim[0]):
#             for j in [k for k, _ in enumerate(cats) if k != cov_idx]:

#                 cf_idx = int(sub_idx_cf_tensor[i][j])
#                 if all(adata[cf_idx].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):

#                     sub_idx_cf_tensor[i][j] = i

#         torch.save(sub_idx_cf_tensor, sub_idx_cf_tensor_path)

#     try:
#         model = FairVI.load(f"{pre_path}/{sub_model_name}", adata=adata_sub)
#     except:
#         adata_sub = adata_sub.copy()
#         FairVI.setup_anndata(
#             adata_sub,
#             layer='counts',
#             categorical_covariate_keys=cats,
#             continuous_covariate_keys=[]
#         )
#         model = FairVI(adata_sub, idx_cf_tensor_path=sub_idx_cf_tensor_path,
#                        cell_to_idx=cell_to_idx, idx_to_cell=idx_to_cell)
#         model.train(**train_dict)
#         try:
#             model.save(f"{pre_path}/{sub_model_name}")
#         except:
#             pass

#     source_adata = adata[source_sub_idx]

#     model.idx_cf_tensor_path = sub_idx_cf_tensor_path
#     model.cell_to_idx = cell_to_source_sub_idx
#     model.idx_to_cell = torch.tensor(source_sub_idx).to(device)

#     px_cf_mean_pred = model.predict_given_covs(adata=source_adata, cats=cats, cov_idx=cov_idx,
#                                                cov_value_cf=cov_value_cf).to('cpu')

#     true_x_count = torch.tensor(adata.layers["counts"][true_idx].toarray())
#     true_x_counts_mean = torch.mean(true_x_count, dim=0).to('cpu')

#     print(f'Counterfactual prediction for {cov_str_cf}, and {other_covs_str}')

#     r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean,
#             px_cf_mean_pred, n_top_deg=n_top_deg)

#     return true_x_counts_mean, px_cf_mean_pred


# def r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean, px_cf_mean_pred, n_top_deg: int = 100):
#     adata.var['name'] = adata.var.index
#     sc.tl.rank_genes_groups(
#         adata, cov_name, method='wilcoxon', key_added="wilcoxon")
#     ranked_genes = sc.get.rank_genes_groups_df(
#         adata, group=cov_value_cf, key='wilcoxon', gene_symbols='name')
#     ranked_genes_names = ranked_genes[ranked_genes['name'].notnull()]['name']
#     deg_names = ranked_genes_names[:n_top_deg]
#     deg_idx = [i for i, _ in enumerate(
#         adata.var['name']) if adata.var['name'][i] in list(deg_names)]

#     r2 = r2_score(true_x_counts_mean, px_cf_mean_pred)

#     r2_log = r2_score(torch.log1p_(true_x_counts_mean),
#                       torch.log1p_(px_cf_mean_pred))

#     r2_deg = r2_score(true_x_counts_mean[deg_idx], px_cf_mean_pred[deg_idx])

#     r2_log_deg = r2_score(torch.log1p_(
#         true_x_counts_mean[deg_idx]), torch.log1p_(px_cf_mean_pred[deg_idx]))

#     print('All Genes')
#     print(f'R2 = {r2:.4f}')
#     print(f'R2 log = {r2_log:.4f}')
#     print(f'DE Genes (n_top={n_top_deg})')
#     print(f'R2 = {r2_deg:.4f}')
#     print(f'R2 log = {r2_log_deg:.4f}')


def pred_ood_old(model,
             adata,  # OOD Adata
             cov_name,
             cov_values: str,
             cov_value_cf: str,
             cats: List[str],
             n_samples: int = 1000,):
    adata.obs['idx'] = [i for i in range(len(adata))]

    true_idx = list(adata[adata.obs[cov_name] == cov_value_cf].obs['idx'])
    source_idx = list(adata[adata.obs[cov_name] == cov_values].obs['idx'])

    true_adata = adata[adata.obs['idx'].isin(true_idx)].copy()
    source_adata = adata[adata.obs['idx'].isin(source_idx)].copy()

    adata_cf = source_adata.copy()

    adata_cf.obs[cov_name] = pd.Categorical(
        [cov_value_cf for _ in adata_cf.obs[cov_name]])
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.setup_anndata(
        adata_cf,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    adata_cf = model._validate_anndata(adata_cf)

    # print("Data loader OOD")
    scdl = model._make_data_loader(
        adata=adata_cf, batch_size=batch_size
    )

    cov_idx = cats.index(cov_name)
    px_cf_mean_list = []
    for tensors in scdl:
        # print("This is one batch ########################")
        px_cf = model.module.sub_forward(idx=cov_idx + 1, x=tensors[REGISTRY_KEYS.X_KEY].to(device),
                                         cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY].to(device))
        # px_cf_mean_list.append(px_cf.sample())
        # px_cf_mean_list.append(px_cf.mean)
        samples = []
        # print("Doing samples ########################")
        for _ in range(n_samples):
            samples.append(px_cf.sample().to('cpu'))
        # print("Samples done ########################")
        samples = torch.stack(samples, dim=0)
        px_cf = torch.mean(samples, dim=0)
        px_cf_mean_list.append(px_cf)

    px_cf_mean_tensor = torch.cat(px_cf_mean_list, dim=0)
    px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)

    px_cf_variance = torch.sub(px_cf_mean_tensor, px_cf_mean_pred)
    px_cf_variance = torch.pow(px_cf_variance, 2)
    px_cf_variance_pred = torch.mean(px_cf_variance, dim=0)

    px_cf_mean_pred, px_cf_variance_pred = px_cf_mean_pred.to(
        'cpu'), px_cf_variance_pred.to('cpu')
    px_cf_mean_tensor = px_cf_mean_tensor.to('cpu').numpy()

    px_cf_mean_tensor = ad.AnnData(px_cf_mean_tensor)
    sc.pp.normalize_total(px_cf_mean_tensor)
    px_cf_mean_tensor = torch.tensor(px_cf_mean_tensor.X)
    sc.pp.normalize_total(true_adata)

    true_x_count = torch.tensor(true_adata.X.toarray())  # X is normalized
    # cf_x_count = torch.tensor(source_adata.X.toarray())

    x_true = true_x_count
    x_pred = px_cf_mean_tensor
    # x_ctrl = cf_x_count

    x_true = np.log1p(x_true)
    x_pred = np.log1p(x_pred)
    # x_ctrl = np.log1p(x_ctrl)
    return x_true, x_pred



def pred_ood(model,
             adata,  # OOD Adata
             cov_name,
             cov_values: str,
             cov_value_cf: str,
             cats: List[str],):
    adata.X = adata.layers['counts']
    adata.obs['idx'] = [i for i in range(len(adata))]

    true_idx = list(adata[adata.obs[cov_name] == cov_value_cf].obs['idx'])
    source_idx = list(adata[adata.obs[cov_name] == cov_values].obs['idx'])

    true_adata = adata[adata.obs['idx'].isin(true_idx)].copy()
    source_adata = adata[adata.obs['idx'].isin(source_idx)].copy()

    adata_cf = source_adata.copy()

    adata_cf.obs[cov_name] = pd.Categorical(
        [cov_value_cf for _ in adata_cf.obs[cov_name]])
    batch_size = 512
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

    cov_idx = cats.index(cov_name)
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
        for px_cf in pxs_cf:
            if px_cf is None:
                continue
            x_cf = px_cf.mu
            px_cf_mean_list.append(x_cf)

    px_cf_mean_tensor = torch.stack(px_cf_mean_list, dim=0)
    px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)

    px_cf_mean_pred = px_cf_mean_pred.to('cpu').detach().numpy()
    px_cf_mean_tensor = px_cf_mean_tensor.to('cpu').detach().numpy()

    px_cf_mean_tensor = ad.AnnData(px_cf_mean_pred)
    px_cf_mean_tensor = torch.tensor(px_cf_mean_tensor.X)

    true_x_count = torch.tensor(true_adata.X.toarray())
    # cf_x_count = torch.tensor(source_adata.X.toarray())

    x_true = true_x_count
    x_pred = px_cf_mean_tensor
    # x_ctrl = cf_x_count

    x_true = np.log1p(x_true)
    x_pred = np.log1p(x_pred)
    # x_ctrl = np.log1p(x_ctrl)
    return x_true, x_pred


def ood_r2_eval(model,
                adata,  # OOD Adata
                cov_name,
                cov_values: str,
                cov_value_cf: str,
                cats: List[str],
                cell_type='Immune (T cell)',):
    x_true, x_pred = pred_ood(model, adata, cov_name, cov_values,
                              cov_value_cf, cats)
    deg_list = adata.uns['rank_genes_groups_ood'][f'{cell_type}_male']

    r2_results = {}
    for n_top_deg in [20, 50, None]:
        if n_top_deg is not None:
            degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            degs = np.arange(adata.n_vars)
            n_top_deg = 'all'

        x_true_deg = x_true[:, degs]
        x_pred_deg = x_pred[:, degs]
        # x_ctrl_deg = x_ctrl[:, degs]
        r2_mean_deg = pearsonr(x_true_deg.mean(0), x_pred_deg.mean(0))[0] # Zero index is the value, 1 is the p-value
        r2_variance_deg = pearsonr(x_true_deg.var(0), x_pred_deg.var(0))[0] # Zero index is the value, 1 is the p-value
        # r2_mean_base_deg = r2_score(x_true_deg.mean(0), x_ctrl_deg.mean(0))

        r2_results[str(n_top_deg)] = {}
        r2_results[str(n_top_deg)]['Mean'] = r2_mean_deg
        r2_results[str(n_top_deg)]['Variance'] = r2_variance_deg
        # r2_results[str(n_top_deg)]['Control'] = r2_mean_base_deg

    r2_results = pd.DataFrame.from_dict(r2_results).T
    return r2_results


def ood_evaluate_emd(model,
                     adata,  # OOD Adata
                     cov_name,
                     cov_values: str,
                     cov_value_cf: str,
                     cats: List[str],
                     n_samples: int = 1000,):
    x_true, x_pred = pred_ood(model, adata, cov_name, cov_values,
                              cov_value_cf, cats, n_samples=n_samples)
    deg_list = adata.uns['rank_genes_groups_ood']['Immune (T cell)_male']

    emd_results = {}
    for n_top_deg in [20, 50, None]:
        if n_top_deg is not None:
            degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
        else:
            train_sub_idx.append(i)

        cell_to_idx[i] = len(train_sub_idx) - 1
        cell_to_source_sub_idx[i] = len(source_sub_idx) - 1

    idx_to_cell = torch.tensor(train_sub_idx).to(device)
    adata_sub = adata[train_sub_idx]

    cov_str_cf = cats[cov_idx] + ' = ' + str(cov_value) + ' to ' + str(cov_value_cf)
    other_covs_str = ', '.join(c + ' = ' + str(other_covs_values[k]) for k, c in enumerate(other_cats))

    app_str = f'--cf--' + cov_str_cf + ', ' + other_covs_str
    sub_model_name = new_model_name + app_str
    sub_idx_cf_tensor_path = idx_cf_tensor_path + app_str + '.pt'

    try:
        sub_idx_cf_tensor = torch.load(sub_idx_cf_tensor_path)
    except:
        path = idx_cf_tensor_path
        if not path.endswith('.pt'):
            path += '.pt'
        sub_idx_cf_tensor = torch.load(path)
        dim = sub_idx_cf_tensor.size()

        for i in range(dim[0]):
            for j in [k for k, _ in enumerate(cats) if k != cov_idx]:

                cf_idx = int(sub_idx_cf_tensor[i][j])
                if all(adata[cf_idx].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):

                    sub_idx_cf_tensor[i][j] = i

        torch.save(sub_idx_cf_tensor, sub_idx_cf_tensor_path)

    try:
        model = FairVI.load(f"{pre_path}/{sub_model_name}", adata=adata_sub)
    except:
        adata_sub = adata_sub.copy()
        FairVI.setup_anndata(
            adata_sub,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )
        model = FairVI(adata_sub, idx_cf_tensor_path=sub_idx_cf_tensor_path,
                       cell_to_idx=cell_to_idx, idx_to_cell=idx_to_cell)
        model.train(**train_dict)
        try:
            model.save(f"{pre_path}/{sub_model_name}")
        except:
            pass

    source_adata = adata[source_sub_idx]

    model.idx_cf_tensor_path = sub_idx_cf_tensor_path
    model.cell_to_idx = cell_to_source_sub_idx
    model.idx_to_cell = torch.tensor(source_sub_idx).to(device)

    px_cf_mean_pred = model.predict_given_covs(adata=source_adata, cats=cats, cov_idx=cov_idx,
                                               cov_value_cf=cov_value_cf).to('cpu')

    true_x_count = torch.tensor(adata.layers["counts"][true_idx].toarray())
    true_x_counts_mean = torch.mean(true_x_count, dim=0).to('cpu')

    print(f'Counterfactual prediction for {cov_str_cf}, and {other_covs_str}')

    r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean, px_cf_mean_pred, n_top_deg=n_top_deg)

    return true_x_counts_mean, px_cf_mean_pred


def r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean, px_cf_mean_pred, n_top_deg: int = 100):
    adata.var['name'] = adata.var.index
    sc.tl.rank_genes_groups(adata, cov_name, method='wilcoxon', key_added="wilcoxon")
    ranked_genes = sc.get.rank_genes_groups_df(adata, group=cov_value_cf, key='wilcoxon', gene_symbols='name')
    ranked_genes_names = ranked_genes[ranked_genes['name'].notnull()]['name']
    deg_names = ranked_genes_names[:n_top_deg]
    deg_idx = [i for i, _ in enumerate(adata.var['name']) if adata.var['name'][i] in list(deg_names)]

    r2 = r2_score(true_x_counts_mean, px_cf_mean_pred)

    r2_log = r2_score(torch.log1p_(true_x_counts_mean), torch.log1p_(px_cf_mean_pred))

    r2_deg = r2_score(true_x_counts_mean[deg_idx], px_cf_mean_pred[deg_idx])

    r2_log_deg = r2_score(torch.log1p_(true_x_counts_mean[deg_idx]), torch.log1p_(px_cf_mean_pred[deg_idx]))

    print('All Genes')
    print(f'R2 = {r2:.4f}')
    print(f'R2 log = {r2_log:.4f}')
    print(f'DE Genes (n_top={n_top_deg})')
    print(f'R2 = {r2_deg:.4f}')
    print(f'R2 log = {r2_log_deg:.4f}')

