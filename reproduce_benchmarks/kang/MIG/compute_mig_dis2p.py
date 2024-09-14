import gc
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
os.chdir('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()

ood_cts = []
for key in adata.obs.keys():
    if ('split' in key) and (len(key.split('_')) == 2):
        ood_cts.append(key.split('_')[1])

del adata
gc.collect()

for cell_type_to_check in ood_cts:
    gc.collect()
    split_key = f'split_{cell_type_to_check}'

    dis2p_model_path = (
        f'kang_dis2p_cE_split_{cell_type_to_check}/'
        f'pretrainAE_0_maxEpochs_1000_split_split_{cell_type_to_check}_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2'
    )
    
    pre_path = '../models/'
    adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad')
    adata = adata[adata.X.sum(1) != 0].copy()
    adata = adata[adata.obs[split_key] == 'valid'].copy()
    
    model = dvi.Dis2pVI_cE.load(f"{pre_path}/{dis2p_model_path}", adata=adata)
    cats = ['cell_type', 'condition']

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
    MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
    results = [MI, MI_not_max, MI_not_min, MI_not, MI_dif_max, MI_dif_min, MI_dif, maxMIG, concatMIG, minMIG]

    output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/kang/MIG/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f"{output_path}/{module_name}_{split_key}_MI_results.pkl", "wb") as f:
        pickle.dump(results, f)
