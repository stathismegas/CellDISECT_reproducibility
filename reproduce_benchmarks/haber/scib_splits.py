import scvi
scvi.settings.seed = 0
import scanpy as sc
import torch
import numpy as np
import pandas as pd
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)
import gc

from dis2p import dis2pvi_cE as dvi
import biolord
from scDisInFact import scdisinfact, create_scdisinfact_dataset

from scib_metrics.benchmark import (
    BatchCorrection,
    Benchmarker,
    BioConservation,
)
import pickle

import warnings
import gc
warnings.filterwarnings("ignore")


RANDOM_SEED = 42

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/haber_hvg_split.h5ad')
adata_biolord = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/haber_hvg_split.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()
adata_biolord = adata_biolord[adata_biolord.layers['counts'].sum(1) != 0].copy()

cats = ['batch', 'condition', 'cell_label',]
condition_key = ['batch', 'condition', 'cell_label',]

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

gc.collect()

from typing import NamedTuple

class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    LATENT_MODE_KEY: str = "latent_mode"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()


cov_names = ['condition']
cov_values = ['Control']
cov_values_cf = ['Salmonella']

n_samples_from_source_max = 500

pre_path = '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/'

def load_models(
    dis2p_model_path: str,
    biolord_model_path: str,
    scdisinfact_model_path: str,
):
    model = dvi.Dis2pVI_cE.load(f"{pre_path}/{dis2p_model_path}", adata=adata)
    biolord_model = biolord.Biolord.load(f"{pre_path}/{biolord_model_path}", adata=adata_biolord)
    
    scdisinfact_model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                        reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                        reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
    scdisinfact_model.load_state_dict(torch.load(f"{pre_path}/{scdisinfact_model_path}", map_location = device))
    
    return model, biolord_model, scdisinfact_model

scenario = 'allOut'
cell_types = list(adata.obs['cell_label'].unique()) # ['Enterocyte.Progenitor', 'Stem', 'TA.Early', 'TA', 'Tuft', 'Enterocyte', 'Goblet', 'Endocrine']
conds = ['salmonella', 'hpoly10']

for cell_type_to_check in cell_types:
    for cond_path_name in conds: 
        gc.collect()
        
        dis2p_model_path = (
            f'dis2p_cE_split_{scenario}_{cell_type_to_check}_{cond_path_name}/'
            f'pretrainAE_0_maxEpochs_1000_split_split_{scenario}_{cell_type_to_check}_{cond_path_name}_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2'
        )
        biolord_model_path = f'biolord/haber_biolord_earlierStop_basicSettings_nb_split_{scenario}_{cell_type_to_check}_{cond_path_name}/'
        scdisinfact_model_path = f'scDisInfact/haber_scdisinfact_defaultSettings_fsplit_{scenario}_{cell_type_to_check}_{cond_path_name}.pth'

        model, biolord_model, scdisinfact_model = load_models(dis2p_model_path, biolord_model_path, scdisinfact_model_path)
        

        # Z_0
        adata.obsm[f'dis2p_cE_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

        for i in range(len(cats)):
            null_idx = [s for s in range(len(cats)) if s != i]
            # Z_i
            adata.obsm[f'dis2p_cE_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
            # Z_{-i}
            adata.obsm[f'dis2p_cE_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)
            
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
            adata.obsm[f'Biolord_Z_not_{i+1}'] = latent_adata.X.copy()

        adata.obsm[f'Biolord_Z_0'] = latent_unknown_attributes_adata.X.copy()

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
            
        for label_key in cats:
            adata.obs[label_key] = adata.obs[label_key].astype('category')

        bms = {}
        batch_correction_metrics = BatchCorrection(silhouette_batch=False)

        for label_key in cats:
            print(f"Label: {label_key}")
            bms[label_key] = {}
            label_ind = cats.index(label_key) + 1
            embedding_obsm_keys = [f'dis2p_cE_Z_{label_ind}', 
                                f'Biolord_Z_{label_ind}',
                                f'scDisInfact_Z_{label_ind}',]
            for batch_key in cats:
                if batch_key == label_key:
                    continue
                print(f"Batch: {batch_key}")
                bm = Benchmarker(
                    adata,
                    batch_key=batch_key,
                    label_key=label_key,
                    embedding_obsm_keys=embedding_obsm_keys,
                    batch_correction_metrics=batch_correction_metrics,
                    n_jobs=-1,
                    )
                bm.benchmark()
                bms[label_key][batch_key] = bm

        # Save the results
        with open(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/haber/scib_results/{scenario}_{cell_type_to_check}_{cond_path_name}.pkl', 'wb') as f:
            pickle.dump(bms, f)
