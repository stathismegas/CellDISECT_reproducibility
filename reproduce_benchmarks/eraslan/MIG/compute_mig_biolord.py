import sys
split_key = sys.argv[1]

import os
import numpy as np
import scvi
scvi.settings.seed = 0
import scanpy as sc
import torch
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

import biolord

import pickle
import warnings

sys.path.insert(1, '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
os.chdir('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

if split_key in ['split_1', 'split_2']:
    cats = ['tissue', 'Sample ID', 'sex', 'Age_bin']
elif split_key in ['split_4']:
    cats = ['tissue', 'Sample ID', 'Age_bin']
else:
    raise ValueError(f"split_key {split_key} not recognized")
pre_path = '../models/'

biolord_model_path = f'biolord/eraslan_biolord_NoCT_earlierStop_basicSettings_nb_{split_key}/'

biolord_model = biolord.Biolord.load(f"{pre_path}/{biolord_model_path}", adata=adata)


for i, cat in enumerate(cats):
    nullify = [c for c in cats if c != cat]
    latent_unknown_attributes_adata, latent_adata = biolord_model.get_latent_representation_adata(
        adata=adata,
        nullify_attribute=nullify,
    )
    adata.obsm[f'Biolord_Z_{i+1}'] = latent_adata.X.copy()

    nullify = [cat]
    latent_unknown_attributes_adata, latent_adata = biolord_model.get_latent_representation_adata(
        adata=adata,
        nullify_attribute=nullify,
    )
    adata.obsm[f'Biolord_Z_not_{i+1}'] = latent_adata.X.copy()

adata.obsm[f'Biolord_Z_0'] = latent_unknown_attributes_adata.X.copy()

adata = adata[adata.obs[split_key] == 'val'].copy()

create_cats_idx(adata, cats)
module_name = "Biolord"
MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
results = [MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG]

output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/eraslan/MIG/results'
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(f"{output_path}/{module_name}_{split_key}_MI_results.pkl", "wb") as f:
    pickle.dump(results, f)
