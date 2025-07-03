import sys
split_key = sys.argv[1]

import os

import scvi
scvi.settings.seed = 0
import scanpy as sc
import torch
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from celldisect import CellDISECT

import pickle
import warnings

import sys
sys.path.insert(1, '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
os.chdir('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()
adata = adata[adata.obs[split_key] == 'val'].copy()

if split_key in ['split_1', 'split_2']:
    cats = ['tissue', 'Sample ID', 'sex', 'Age_bin']
elif split_key in ['split_4']:
    cats = ['tissue', 'Sample ID', 'Age_bin']
else:
    raise ValueError(f"split_key {split_key} not recognized")

pre_path = '../models/'

celldisect_model_path = (
    f'celldisect_{split_key}/'
    f'pretrainAE_0_maxEpochs_1000_split_{split_key}_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2_batch_size_256_NoCT'
    )

model = CellDISECT.load(f"{pre_path}/{celldisect_model_path}", adata=adata)

# Z_0
adata.obsm[f'celldisect_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

for i in range(len(cats)):
    null_idx = [s for s in range(len(cats)) if s != i]
    # Z_i
    adata.obsm[f'celldisect_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    # Z_{-i}
    adata.obsm[f'celldisect_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)
    
create_cats_idx(adata, cats)
module_name = "celldisect"
MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
results = [MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG]

output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/eraslan/MIG/results'
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(f"{output_path}/{module_name}_{split_key}_MI_results.pkl", "wb") as f:
    pickle.dump(results, f)
