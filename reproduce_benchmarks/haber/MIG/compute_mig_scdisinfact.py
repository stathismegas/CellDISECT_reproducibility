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

from scDisInFact import scdisinfact, create_scdisinfact_dataset
import pandas as pd

import pickle
import warnings

import sys
sys.path.insert(1, '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
os.chdir('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility')
from metrics.metrics import Mixed_KSG_MI_metrics, create_cats_idx

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/haber_hvg_split.h5ad')
adata = adata[adata.X.sum(1) != 0].copy()
ood_cts = list(adata.obs['cell_label'].unique())
del adata
gc.collect()

for cond in ['Salmonella', 'Hpoly.Day10']:
    if cond == 'Salmonella':
        cond_path_name = 'salmonella'
    elif cond == 'Hpoly.Day10':
        cond_path_name = 'hpoly10'

    scenario = 'allOut'
    
    for cell_type_to_check in ood_cts:
        gc.collect()

        split_key = f'{scenario}_{cell_type_to_check}_{cond_path_name}'
        
        adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/haber_hvg_split.h5ad')
        adata = adata[adata.X.sum(1) != 0].copy()
        adata = adata[adata.obs[split_key].isin(['train', 'val'])].copy()
        
        cats = ['batch', 'condition', 'cell_label',]

        pre_path = '../models/'
        scdisinfact_model_path = f'scDisInfact/eraslan_scdisinfact_defaultSettings_f{split_key}.pth'

        condition_key = cats

        #scDisInfact Related Data
        reg_mmd_comm = 1e-4
        reg_mmd_diff = 1e-4
        reg_kl_comm = 1e-5
        reg_kl_diff = 1e-2
        reg_class = 1
        reg_gl = 1
        Ks = [40] + [10] * len(condition_key)
        batch_size = 64
        interval = 10
        lr = 5e-4
        batch_size = 64
        adata_ = adata[adata.layers['counts'].sum(1) != 0].copy()
        counts = adata_.layers['counts'].copy()
        meta_cells = adata_.obs.copy()
        meta_cells['one'] = pd.Categorical([1 for _ in range(adata_.n_obs)])
        data_dict = create_scdisinfact_dataset(counts, meta_cells, condition_key = condition_key, batch_key = "one")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        del adata_


        scdisinfact_model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                            reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                            reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
        scdisinfact_model.load_state_dict(torch.load(f"{pre_path}/{scdisinfact_model_path}", map_location = device))


        # one forward pass
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
            
        adata = adata[adata.obs[split_key] == 'val'].copy()
        create_cats_idx(adata, cats)
        module_name = "scDisInfact"
        MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)
        results = [MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG]

        output_path = f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/haber/MIG/results'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(f"{output_path}/{module_name}_{split_key}_MI_results.pkl", "wb") as f:
            pickle.dump(results, f)
