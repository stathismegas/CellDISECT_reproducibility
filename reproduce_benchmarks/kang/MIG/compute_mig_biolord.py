import gc
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
    biolord_model_path = f'biolord/kang_biolord_earlierStop_basicSettings_nb_split_{cell_type_to_check}/'
    scdisinfact_model_path = f'scDisInfact/kang_scdisinfact_40_10_split_{cell_type_to_check}.pth'    
    
    pre_path = '../models/'
    adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/kang_normalized_hvg.h5ad')
    adata = adata[adata.X.sum(1) != 0].copy()
    biolord_model = biolord.Biolord.load(f"{pre_path}/{biolord_model_path}", adata=adata)

    cats = ['cell_type', 'condition']

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

    adata = adata[adata.obs[split_key] == 'valid'].copy()

    create_cats_idx(adata, cats)
    module_name = "Biolord"
    MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
    results = [MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG]

    output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/kang/MIG/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f"{output_path}/{module_name}_{split_key}_MI_results.pkl", "wb") as f:
        pickle.dump(results, f)
