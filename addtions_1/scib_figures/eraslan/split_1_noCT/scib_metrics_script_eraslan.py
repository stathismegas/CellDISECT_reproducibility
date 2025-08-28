import warnings
import pickle
import scanpy as sc
import scvi
import torch

from scib_metrics.benchmark import Benchmarker

from celldisect import CellDISECT

scvi.settings.seed = 0
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

datapath = '/lustre/scratch126/cellgen/lotfollahi/aa34/celldisect/datasets/eraslan_preprocessed1212_split_deg.h5ad'
adata = sc.read_h5ad(datapath)

adata = adata[adata.layers['counts'].sum(1) != 0].copy()

cats = ['tissue', 'Sample ID', 'Age_bin']

models_base_path = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/models_celldisect/celldisect_split_1'

# Define the 5 CellDISECT models with different latent dimensions
model_folders = {
    'celldisect_8': 'pretrainAE_0_maxEpochs_1000_split_split_1_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_8_n_layers_2batch_size_256_NoCT',
    'celldisect_16': 'pretrainAE_0_maxEpochs_1000_split_split_1_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_16_n_layers_2batch_size_256_NoCT',
    'celldisect_32': 'pretrainAE_0_maxEpochs_1000_split_split_1_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2batch_size_256_NoCT',
    'celldisect_64': 'pretrainAE_0_maxEpochs_1000_split_split_1_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_64_n_layers_2batch_size_256_NoCT',
    'celldisect_128': 'pretrainAE_0_maxEpochs_1000_split_split_1_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_128_n_layers_2batch_size_256_NoCT'
}

# Load all 5 CellDISECT models
adata.obs['_cluster'] = '0'
celldisect_models = {}

for model_name, model_folder in model_folders.items():
    model_path = f"{models_base_path}/{model_folder}"
    print(f"Loading {model_name} from {model_path}")
    celldisect_models[model_name] = CellDISECT.load(model_path, adata=adata)

# get latents of all 5 CellDISECT models
for model_name, model in celldisect_models.items():
    print(f"Extracting latents for {model_name}")
    
    # Z_0: all covariates nullified, shared not nullified
    adata.obsm[f'{model_name}_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

    for i in range(len(cats)):
        null_idx = [s for s in range(len(cats)) if s != i]
        # Z_i: all covariates except i nullified, shared nullified
        adata.obsm[f'{model_name}_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
        # Z_{-i}: only covariate i nullified, shared not nullified
        adata.obsm[f'{model_name}_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)

# ============================== Done Getting Latents ==============================

# Create a subset of data
adata_ = sc.pp.subsample(adata, fraction=0.1, copy=True)    

# Benchmark on all latent spaces and categories for all 5 CellDISECT models
all_bms = []
model_names = list(celldisect_models.keys())

for i in range(len(cats) + 1):
    print(f'Latent Z_{i}')
    # Create embedding keys for all 5 models
    embedding_obsm_keys = [f'{model_name}_Z_{i}' for model_name in model_names]

    print('embedding_obsm_keys: ', embedding_obsm_keys)
    bms = {}
    for batch_key in cats:
        if (i > 0) and (batch_key == cats[i-1]):
            continue
        print(f'Processing batch_key: {batch_key}')
        bm = Benchmarker(
            adata_,
            batch_key=batch_key,
            label_key='Broad cell type',
            embedding_obsm_keys=embedding_obsm_keys,
            n_jobs=-1,
        )
        bm.benchmark()
        bms[batch_key] = bm
    all_bms.append(bms)

# Save results
output_path = '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/models_celldisect/scib_results_celldisect_latent_dims.pkl'
with open(output_path, 'wb') as handle:
    pickle.dump(all_bms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f"Results saved to: {output_path}")
