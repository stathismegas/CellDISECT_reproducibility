import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Union
import random
import pickle
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import scvi
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

from scvi.model.base import BaseModelClass
from scib_metrics.benchmark import (
    BatchCorrection,
    Benchmarker,
    BioConservation,
)

import biolord
from celldisect import CellDISECT
from scDisInFact import scdisinfact, create_scdisinfact_dataset

scvi.settings.seed = 0
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

datapath = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/HLCA_preprocessed_for_CellDISECT.h5ad'
adata = sc.read_h5ad(datapath)
adata_biolord = sc.read_h5ad(datapath)

adata = adata[adata.layers['counts'].sum(1) != 0].copy()
adata_biolord = adata_biolord[adata_biolord.layers['counts'].sum(1) != 0].copy()

cats = ['donor_id', 'sex', 'age_category', 'tissue']

pre_path = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/training'
celldisect_model_path = '/celldisect_random_split_/HLCA_weighted_clf_False_pretrainAE_0_maxEpochs_1000_random_split_reconW_20_cfWeight_0.8_beta_0.003_clf_0.02_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT/'
biolord_model_path = '/biolord/HLCA_biolord_NoCT_earlierStop_basicSettings_nb_random_split/'
scdisinfact_model_path = f'/scdisinfact/HLCA_scdisinfact_NoCT_defaultSettings_random_split.pth'

# Load CellDISECT
adata.obs['_cluster'] = '0'
celldisect_model = CellDISECT.load(f"{pre_path}/{celldisect_model_path}", adata=adata)

#Load Biolord
biolord_model = biolord.Biolord.load(f"{pre_path}/{biolord_model_path}", adata=adata_biolord)

# Load scDisInfact
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [40, 10, 10, 10, 10]
batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4

lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
batch_size = 64
adata_ = adata[adata.layers['counts'].sum(1) != 0].copy()
counts = adata_.layers['counts'].copy()
meta_cells = adata_.obs.copy()
meta_cells['one'] = pd.Categorical([1 for _ in range(adata_.n_obs)])

data_dict = create_scdisinfact_dataset(counts, meta_cells, condition_key = cats, batch_key = "one")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scdisinfact_model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                    reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
scdisinfact_model.load_state_dict(torch.load(f"{pre_path}/{scdisinfact_model_path}", map_location = device))

# get latents of CellDISECT
adata.obsm[f'celldisect_Z_0'] = celldisect_model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

for i in range(len(cats)):
    null_idx = [s for s in range(len(cats)) if s != i]
    # Z_i
    adata.obsm[f'celldisect_Z_{i+1}'] = celldisect_model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    # Z_{-i}
    adata.obsm[f'celldisect_Z_not_{i+1}'] = celldisect_model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)
    
    
# get latents of Biolord
for i, cat in enumerate(cats):
    nullify = [c for c in cats if c != cat]
    latent_unknown_attributes_adata, latent_adata = biolord_model.get_latent_representation_adata(
        adata=adata_biolord,
        nullify_attribute=nullify,
    )
    adata.obsm[f'Biolord_Z_{i+1}'] = latent_adata.X.copy()

    nullify = [cat]
    latent_unknown_attributes_adata, latent_adata = biolord_model.get_latent_representation_adata(
        adata=adata_biolord,
        nullify_attribute=nullify,
    )
    adata.obsm[f'Biolord_Z_not_{i+1}'] = np.concatenate([latent_adata.X.copy()], axis=1)

adata.obsm[f'Biolord_Z_0'] = latent_unknown_attributes_adata.X.copy()


# get latents of scDisInfact
z_cs = []
z_ds = []
zs = []

# loop through all training count matrices
for dataset in data_dict["datasets"]:
    with torch.no_grad():
        # pass through the encoders
        dict_inf = scdisinfact_model.inference(counts = dataset.counts_norm.to(scdisinfact_model.device), batch_ids = dataset.batch_id[:,None].to(scdisinfact_model.device), print_stat = False)
        # pass through the decoder
        dict_gen = scdisinfact_model.generative(z_c = dict_inf["mu_c"], z_d = dict_inf["mu_d"], batch_ids = dataset.batch_id[:,None].to(scdisinfact_model.device))
        z_c = dict_inf["mu_c"]
        z_d = dict_inf["mu_d"]
        mu = dict_gen["mu"]    
        z_ds.append([x.cpu().detach().numpy() for x in z_d])
        z_cs.append(z_c.cpu().detach().numpy())

# shared-bio factor, concatenate across all training matrices
z_cs = np.concatenate(z_cs, axis = 0)
adata.obsm[f'scDisInfact_Z_0'] = z_cs.copy()

# unshared-bio factors for conditions 1 and 2
for i, cat in enumerate(cats):
    z_ds_cond = np.concatenate([x[i] for x in z_ds], axis = 0)
    adata.obsm[f'scDisInfact_Z_{i+1}'] = z_ds_cond.copy()

for i in range(1, len(cats)+1):
    
    z_not_ds_cond = np.concatenate([adata.obsm[f'scDisInfact_Z_{j}'].copy() for j in range(len(cats)+1) if j != i], axis=1)
    adata.obsm[f'scDisInfact_Z_not_{i}'] = z_not_ds_cond.copy()
    
# ============================== Done Getting Latents ==============================

# Only the pure bio conserving latents
embedding_obsm_keys = [f'celldisect_Z_0',
                       f'Biolord_Z_0',
                       f'scDisInfact_Z_0',]

# Create a subset of data
adata_ = sc.pp.subsample(adata, fraction=0.1, copy=True)    
    
# Benchamrk on all latent spaces and categories and plot them all (4*5=20 plots)
all_bms = []
for i in range(len(cats) + 1):
    print(f'Latent Z_{i}')
    embedding_obsm_keys = [f'celldisect_Z_{i}',
                           f'Biolord_Z_{i}',
                           f'scDisInfact_Z_{i}',]

    bms = {}
    for batch_key in cats:
        if (i > 0) and (batch_key == cats[i-1]):
            continue
        print(batch_key)
        bm = Benchmarker(
        adata_,
        batch_key=batch_key,
        label_key='cell_type',
        embedding_obsm_keys=embedding_obsm_keys,
        n_jobs=-1,
        )
        bm.benchmark()
        bms[batch_key] = bm
    all_bms.append(bms)
    
    
with open('/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/scib_figures/all_bms_hlca_ct.pkl', 'wb') as handle:
    pickle.dump(all_bms, handle, protocol=pickle.HIGHEST_PROTOCOL)