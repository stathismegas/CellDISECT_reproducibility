import os

import scvi
scvi.settings.seed = 0
import scanpy as sc
import torch
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from dis2p import dis2pvi_cE as dvi

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

dis2p_model_path = 'dis2p_cE_split_2/' + 'weightedCLF_split_split_2_cfWeight_0.7873_beta_0.003_clf_0.8149_adv_0.0138_advp_5_n_cf_1_lr_0.001_wd_7.78e-05_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32/'
# dis2p_model_path = 'dis2p_cE_split_2/' + 'pretrainAE_30_350_split_split_2_reconW_28.55_cfWeight_0.7873_beta_0.003_clf_0.8149_adv_0.0138_advp_5_n_cf_1_lr_0.001_wd_7.78e-05_new_cf_True_dropout_0.2_n_hidden_128_n_latent_32_n_layers_2/'

model = dvi.Dis2pVI_cE.load(f"{pre_path}/{dis2p_model_path}", adata=adata)

# Z_0
adata.obsm[f'dis2p_cE_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

for i in range(len(cats)):
    null_idx = [s for s in range(len(cats)) if s != i]
    # Z_i
    adata.obsm[f'dis2p_cE_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    # Z_{-i}
    adata.obsm[f'dis2p_cE_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)
    

create_cats_idx(adata, cats)
module_name = "dis2p_cE"

MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
results = [MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG]

# save the results to a file
output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/eraslan/MIG/results'
with open(f"{output_path}/{module_name}_MI_results.pkl", "wb") as f:
    pickle.dump(results, f)
