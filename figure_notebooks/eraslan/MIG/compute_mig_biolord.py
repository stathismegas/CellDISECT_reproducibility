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

import sys
sys.path.insert(1, '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('../eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()
sc.pp.subsample(adata, fraction=0.1)

cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']
pre_path = '../models/'

biolord_model_path = 'biolord/eraslan_biolord_basicSettings_nb_split_2/'

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
    adata.obsm[f'Biolord_Z_not_{i+1}'] = np.concatenate([latent_unknown_attributes_adata.X.copy(), latent_adata.X.copy()], axis=1)

adata.obsm[f'Biolord_Z_0'] = latent_unknown_attributes_adata.X.copy()


create_cats_idx(adata, cats)
module_name = "Biolord"

MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
results = [MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG]

# save the results to a file
output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/eraslan/MIG/results'
with open(f"{output_path}/{module_name}_MI_results.pkl", "wb") as f:
    pickle.dump(results, f)
