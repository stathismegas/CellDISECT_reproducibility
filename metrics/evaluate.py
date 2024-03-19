from typing import List, Tuple
import numpy as np
import torch
from sklearn.metrics import r2_score
from anndata import AnnData
import scanpy as sc
import itertools

from scfair.fairvi import FairVI

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def latent(
        adata: AnnData,
        cats: List[str],
        new_model_name='',
        pre_path: str = '.',
        idx_cf_tensor_path: str = '',
        plot_umap: bool = True,
        **train_dict,
):
    try:
        model = FairVI.load(f"{pre_path}/{new_model_name}", adata=adata)

    except:

        FairVI.setup_anndata(
            adata,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )
        model = FairVI(adata, idx_cf_tensor_path=idx_cf_tensor_path)
        model.train(**train_dict)
        model.save(f"{pre_path}/{new_model_name}")

    model.idx_cf_tensor_path = idx_cf_tensor_path

    print(f"---latent 0 computation started---")
    latent = model.get_latent_representation(nullify_cat_covs_indices=list(range(len(cats))))
    adata.obsm[f"Z_0"] = latent

    for c in range(len(cats)):
        print(f"---latent {c + 1} computation started---")
        null_idx = [s for s in range(len(cats)) if s != c]
        latent = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
        adata.obsm[f"Z_{c + 1}"] = latent

        # concat all Z except c
        null_idx = [c]
        latent = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=False)
        adata.obsm[f"Z_not_{c + 1}"] = latent

    print(f"---latent computation completed---")

    if plot_umap:

        print("---UMAP computation started---")

        for i in range(len(cats) + 1):  # loop over Z_shared and all Z_i

            latent_name = f'Z_{i}'

            print(f"---UMAP for {latent_name}---")

            sc.pp.neighbors(adata, use_rep=f"{latent_name}")
            sc.tl.umap(adata)

            sc.pl.umap(
                adata,
                color=cats,
                ncols=len(cats),
                frameon=False,
            )

        print("---UMAP computation completed---")

    return model, adata


def scib_benchmark(Benchmarker, adata, cats, batch_key_idx, label_key_idx):
    batch_key = cats[batch_key_idx]
    label_key = cats[label_key_idx]

    print(f'scib benchmark metrics for (batch_key={batch_key}, label_key={label_key})')

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=[f"Z_{c}" for c in range(len(cats) + 1)],
        n_jobs=-1,
    )
    bm.benchmark()

    bm.plot_results_table(min_max_scale=False)


def ood_for_given_covs_1(
        adata: AnnData,
        cats: List[str],
        new_model_name='',
        pre_path: str = '.',
        idx_cf_tensor_path: str = '',
        cov_idx: int = 0,
        cov_value_idx: int = 1,
        cov_value_cf_idx: int = 0,
        other_covs_values: Tuple = (0,),
        n_top_deg: int = 100,
        **train_dict,
):
    # remove cells with other_covs_values and cov_value_cf

    cov_name = cats[cov_idx]
    cov_value = tuple(set(adata.obs[cats[cov_idx]]))[cov_value_idx]
    cov_value_cf = tuple(set(adata.obs[cats[cov_idx]]))[cov_value_cf_idx]

    other_cats = [c for c in cats if c != cov_name]

    train_sub_idx = []
    true_idx = []
    source_sub_idx = []
    cell_to_idx = torch.tensor(list(range(adata.n_obs))).to(device)
    cell_to_source_sub_idx = torch.tensor(list(range(adata.n_obs))).to(device)
    for i in range(adata.n_obs):
        cov_value_i = adata[i].obs[cats[cov_idx]][0]

        if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
            if cov_value_i == cov_value_cf:
                true_idx.append(i)
            else:
                train_sub_idx.append(i)
                if cov_value_i == cov_value:
                    source_sub_idx.append(i)
        else:
            train_sub_idx.append(i)

        cell_to_idx[i] = len(train_sub_idx) - 1
        cell_to_source_sub_idx[i] = len(source_sub_idx) - 1

    idx_to_cell = torch.tensor(train_sub_idx).to(device)

    adata_sub = adata[train_sub_idx]

    cov_str_cf = cats[cov_idx] + ' = ' + str(cov_value) + ' to ' + str(cov_value_cf)
    other_covs_str = ', '.join(c + ' = ' + str(other_covs_values[k]) for k, c in enumerate(other_cats))

    app_str = f' cf ' + cov_str_cf + ', ' + other_covs_str
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
            for j in range(dim[1]):
                cf_idx = int(sub_idx_cf_tensor[i][j])
                if all(adata[cf_idx].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)) and \
                        (adata[cf_idx].obs[cats[cov_idx]][0] == cov_value_cf):
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


def ood_for_given_covs_2(
        adata: AnnData,
        cats: List[str],
        new_model_name='',
        pre_path: str = '.',
        idx_cf_tensor_path: str = '',
        cov_idx: int = 0,
        cov_value: str = '',
        cov_value_cf: str = '',
        other_covs_values: Tuple = (0,),
        n_top_deg: int = 100,
        **train_dict,
):
    # remove all cells with other_covs_values

    cov_name = cats[cov_idx]

    other_cats = [c for c in cats if c != cov_name]

    train_sub_idx = []
    true_idx = []
    source_sub_idx = []
    cell_to_idx = torch.tensor(list(range(adata.n_obs))).to(device)
    cell_to_source_sub_idx = torch.tensor(list(range(adata.n_obs))).to(device)

    for i in range(adata.n_obs):
        cov_value_i = adata[i].obs[cats[cov_idx]][0]

        if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
            if cov_value_i == cov_value_cf:
                true_idx.append(i)
            elif cov_value_i == cov_value:
                source_sub_idx.append(i)
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

