import sys
name = sys.argv[1]

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'

import scvi
scvi.settings.seed = 0
import scanpy as sc
import anndata as ad
import torch
import numpy as np
import pandas as pd
import json
#from datetime import datetime
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

import dis2p.dis2pvi_cE as dvi

import rapids_singlecell as rsc
import cupy as cp

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(
    managed_memory=True,  # Allows oversubscription
    pool_allocator=False,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)

from cuml.decomposition import PCA
import anndata as ad
import gc

print("Imports done, starting the script...")

model_names = {
    'optimal': 'maxEpoch_1000_reconW_28.55_cfWeight_0.7873_beta_0.003_clf_0.8149_adv_0.0138_advp_5_n_cf_1_lr_0.001_wd_7.78e-05_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32_n_layers_6',
    'no_clf': 'maxEpoch_1000_reconW_28.55_cfWeight_0.7873_beta_0.003_clf_0.0_adv_0.0138_advp_5_n_cf_1_lr_0.001_wd_7.78e-05_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32_n_layers_6',
    'low_clf': 'maxEpoch_1000_reconW_28.55_cfWeight_0.7873_beta_0.003_clf_0.08_adv_0.0138_advp_5_n_cf_1_lr_0.001_wd_7.78e-05_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32_n_layers_6'
}

output_folder = '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/antony/latents/'

root_dir = '/lustre/scratch126/cellgen/team298/ar32/farm_job_objects_and_outputs_temp/new_dis2p_20240619/prepare_data/'
data = 'dis2p_prepped_20240619'

print(f"Loading the dataset...")
adata = sc.read(root_dir + data + '.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

def create_cats_idx(adata, cats):
    # create numerical index for each attr in cats

    for i in range(len(cats)):
        values = list(set(adata.obs[cats[i]]))

        val_to_idx = {v: values.index(v) for v in values}

        idx_list = [val_to_idx[v] for v in adata.obs[cats[i]]]

        adata.obs[cats[i] + '_idx'] = pd.Categorical(idx_list)

    return adata

# specify attributes
cats = ['integration_donor', 'integration_biological_unit', 'integration_sample_status', 'integration_library_platform_coarse',
        'organ','bin_age']

# create numerical index for each attr in cats
create_cats_idx(adata, cats)

module_name = f'Antony'
pre_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/{module_name}'

print(f"Loading the model...")
model_name = model_names[name]
# load model
model = dvi.Dis2pVI_cE.load(f"{pre_path}/{model_name}", adata=adata)

print(f"Getting the latent 0...")
# Z_0
adata.obsm[f'dis2p_cE_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

for i in range(len(cats)):
    print(f"Getting the latent {i+1} / {len(cats)}...")
    null_idx = [s for s in range(len(cats)) if s != i]
    label = cats[i]
    # Z_i
    adata.obsm[f'dis2p_cE_Z_{label}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    # Z_{-i}
    adata.obsm[f'dis2p_cE_Z_not_{label}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)

cats_to_visualise_in_latent = ['organ','bin_age']

indexes = []
for cat in cats_to_visualise_in_latent:
    indexes = indexes + [cats.index(cat)]

null_idx = [x for x in range(len(cats)) if x not in indexes]

print(f"Getting the latents for {cats_to_visualise_in_latent}...")
# Z_i - latent space that retains information only about cats_to_visualise_in_latent or unsupervised covariates (ie due to Z_shared a.k.a Z_0)
adata.obsm[f'dis2p_cE_Z_{"_".join(cats_to_visualise_in_latent)}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)

print(f"Getting the latents for everything but {cats_to_visualise_in_latent}...")
# Z_{-i} - latent space that retains information about everything but cats_to_visualise_in_latent
adata.obsm[f'dis2p_cE_Z_not_{"_".join(cats_to_visualise_in_latent)}'] = model.get_latent_representation(nullify_cat_covs_indices=indexes, nullify_shared=True)

print("Saving the results part 1...")
adata.write(output_folder + f'/adata_bin_age_with_latents_{name}.h5ad')


n_components = 50

print("Calculating PCA...")
adata.obsm["X_pca"] = PCA(n_components=n_components, output_type="numpy").fit_transform(adata.X)

print("Calculating Neighbors, UMAP for PCA...")
latent = ad.AnnData(X=adata.obsm["X_pca"], obs=adata.obs)
rsc.get.anndata_to_GPU(latent)
rsc.pp.neighbors(adata=latent, n_neighbors = 50)
rsc.tl.umap(adata=latent)
rsc.get.anndata_to_CPU(latent)

adata.uns['control_neighbors'] = latent.uns['neighbors']
adata.uns['control_umap'] = latent.obsm['X_umap']

del latent
gc.collect()

latents_to_plot = ['all'] + cats
counter = 0
for i in range(len(cats) + 1):
    if counter == 0:
        latent_name = f'dis2p_cE_Z_0'
    else:
        latent_name = f'dis2p_cE_Z_{cats[i-1]}'

    print(f"Calculating Neighbors, UMAP for {latent_name}...")
    latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
    rsc.get.anndata_to_GPU(latent)
    rsc.pp.neighbors(adata=latent, n_neighbors = 50)
    rsc.tl.umap(adata=latent)
    rsc.get.anndata_to_CPU(latent)
    
    adata.uns[f'{latent_name}_neighbors'] = latent.uns['neighbors']
    adata.uns[f'{latent_name}_umap'] = latent.obsm['X_umap']

    del latent
    gc.collect()
    counter +=1

i = 'organ_bin_age'
latent_name = f'dis2p_cE_Z_{i}'
print(f"Calculating Neighbors, UMAP for {latent_name}...")
latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
rsc.get.anndata_to_GPU(latent)
rsc.pp.neighbors(adata=latent, n_neighbors = 50)
rsc.tl.umap(adata=latent)
rsc.get.anndata_to_CPU(latent)
adata.uns[f'{latent_name}_neighbors'] = latent.uns['neighbors']
adata.uns[f'{latent_name}_umap'] = latent.obsm['X_umap']

del latent
gc.collect()    

print("Saving the final results...")
adata.write(output_folder + f'/adata_bin_age_with_latents_and_graphs_{name}.h5ad')