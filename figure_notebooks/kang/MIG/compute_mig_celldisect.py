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
sys.path.insert(1, '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/CellDISECT_reproducibility')
os.chdir('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/CellDISECT_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('../kang_normalized_hvg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()

cats = ['cell_type', 'condition']
pre_path = '../models/'

celldisect_model_path = 'kang_celldisect_split_CD4 T/' + 'weightedCLF_split_split_CD4 T_cfWeight_0.5_beta_0.0029_clf_0.4_adv_0.2_advp_3_n_cf_2_lr_0.003_wd_6.00e-05_new_cf_True_dropout_0.2_n_hidden_2048_n_latent_256_n_layers_2/'

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
# sc.pp.subsample(adata, fraction=0.1)
import gc
gc.collect()

MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
results = [MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG]

# save the results to a file
output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/CellDISECT_reproducibility/figure_notebooks/kang/MIG/results'
with open(f"{output_path}/{module_name}_MI_results.pkl", "wb") as f:
    pickle.dump(results, f)
