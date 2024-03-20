import os
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'

import scvi
scvi.settings.seed = 0
import scanpy as sc
import anndata as ad
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.sparse import csr_matrix
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

# import metrics
#from metrics.benchmark_VAE.vi import VI
from metrics.metrics import *
from metrics.diffair_evaluate import *
from metrics.scib_metrics_dev.src.scib_metrics.benchmark import Benchmarker

from dis2p.dis2pvi_cE import Dis2pVI_cE
import biolord
from scDisInFact import scDisInFact


def create_cats_idx(adata, cats):
    # create numerical index for each attr in cats

    for i in range(len(cats)):
        values = list(set(adata.obs[cats[i]]))

        val_to_idx = {v: values.index(v) for v in values}

        idx_list = [val_to_idx[v] for v in adata.obs[cats[i]]]

        adata.obs[cats[i] + '_idx'] = pd.Categorical(idx_list)

    return adata


paths = [
        "data/hca_subsampled_20k.h5ad",
        "data/liver_infected.h5ad",
        "data/kang_2018.h5ad",
        "data/merged_sim_20240318_donor_added.h5ad"
        ]

stubs = [
        "heart",
        "blood",
        "liver",
        "simulation"
        ]

# ---------------- general preprocessing ---------------------
today = datetime.today().strftime('%Y-%m-%d')

epochs = 200
batch_size = 128
cf_weight = 1
beta = 1
clf_weight = 50
adv_clf_weight = 10
adv_period = 1
n_cf = 1

# architecture params
n_layers=1

train_dict = {'max_epochs': epochs, 'batch_size': batch_size, 'cf_weight': cf_weight,
              'beta': beta, 'clf_weight': clf_weight, 'adv_clf_weight': adv_clf_weight,
              'adv_period': adv_period, 'n_cf': n_cf}

module_name = 'dis2p_cE'
pre_path = f"models/{module_name}"

if not os.path.exists(pre_path):
    os.makedirs(pre_path)




for i, path in enumerate(paths):

    adata = sc.read(path)
    # specify name of dataset 
    data_name = stub[i] 
    
     # specify attributes
    if data_name == "heart":
        cats = ['cell_type', 'cell_source', 'gender', 'region']
        Ks = [8, 4, 4, 4, 4]
    elif data_name == "blood":
        cats = ['label', 'cell_type']
        Ks = [8, 4, 4]
    elif data_name == "liver":
        cats = ["coarse_time", "zone", "status_control", "mouse"]
        Ks = [8, 4, 4, 4, 4]
    elif data_name == "simulation":
        cats = ["Donor", "Batch", "Group", "age", "disease"]
        Ks = [8, 4, 4, 4, 4, 4]

        

    # preprocess dataset
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
    )


    # create numerical index for each attr in cats
    create_cats_idx(adata, cats)

    # save adata
    # adata.write_h5ad('data/heart_preprocessed1200.h5ad')


    # ---------------- set train params and train ---------------------

    # specify a name for your model
    model_name =  f'{today},{module_name},{data_name},' + f'n_layers={n_layers},' + ','.join(k + '=' + str(v) for k, v in train_dict.items())

#     # load model (if trained before)
#     try:
#         model = Dis2pVI_cE.load(f"{pre_path}/{model_name}", adata=adata)

#     # trains the model (if not trained before) and save it into: pre_path + model_name
#     except:

    Dis2pVI_cE.setup_anndata(
        adata,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    model = Dis2pVI_cE(adata, n_layers=n_layers)
    model.train(**train_dict)
    model.save(f"{pre_path}/{model_name}")


    # ---------------- calculate latent and umaps ---------------------


    # Z_0
    adata.obsm[f'dis2p_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

    for i in range(len(cats)):
        null_idx = [s for s in range(len(cats)) if s != i]
        # Z_i
        adata.obsm[f'dis2p_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
        # Z_{-i}
        adata.obsm[f'dis2p_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)


    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=FutureWarning)

    for i in range(len(cats) + 1):  # loop over all Z_i

        latent_name = f'dis2p_Z_{i}'

        print(f"---UMAP for {latent_name}---")

        sc.pp.neighbors(adata, use_rep=f"{latent_name}")
        sc.tl.umap(adata)

        sc.pl.umap(
            adata,
            color=cats,
            ncols=len(cats),
            frameon=False,
        )

        
        
    # ---------------- train biolord ---------------------

    model_name = "biolord"+data_name
    
    biolord.Biolord.setup_anndata(
        adata=adata,
        ordered_attributes_keys=[],
        categorical_attributes_keys=cats,
        layer="counts"
    )
    module_params = {
    "decoder_width": 512,
    "decoder_depth": 6,
    "attribute_nn_width": 256,
    "attribute_nn_depth": 2,
    "unknown_attribute_noise_param": 1e0,
    "seed": 42,
    "n_latent_attribute_ordered": 16,
    "n_latent_attribute_categorical": 16,
    "gene_likelihood": "poisson",
    "reconstruction_penalty": 1e1,
    "unknown_attribute_penalty": 1e0,
    "attribute_dropout_rate": 0.1
    }

    model = biolord.Biolord(
        adata=adata,
        n_latent=128,
        model_name=model_name,
        module_params=module_params,
    )

    model.save(f'models/biolord_{data_name}_{today}')

    for i, c in enumerate(cats):
        nullify_attribute = [cat for cat in cats if cat != c]
        _, latent_adata = model.get_latent_representation_adata(adata=adata, nullify_attribute=nullify_attribute)
        adata.obsm[f"biolord_{i+1}"] = latent_adata.X
    if data_name=="blood":
        adata.obsm[f"biolord_cell_type"] = adata.obsm[f"biolord_2"]
    elif data_name=="heart":
        adata.obsm[f"biolord_cell_type"] = adata.obsm[f"biolord_2"]
    elif data_name=="simulation":
        adata.obsm[f"biolord_cell_type"] = adata.obsm[f"biolord_2"]
    

    # ---------------- train scDisInFact ---------------------
    adata.obs['one'] = pd.Categorical([1 for _ in range(adata.n_obs)])
    data_dict = scDisInFact.create_scdisinfact_dataset(adata.layers["counts"], adata.obs, condition_key=cats, batch_key='one')
    
    # training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = scdisinfact(data_dict = data_dict, Ks = Ks, device = device)
    losses = model.train_model(nepochs = 100)
    _ = model.eval()


    # ---------------- train Harmony ---------------------

    from harmony import harmonize
    adata.obsm["Harmony"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key="label")




        
        
        
        
    adata.write_h5ad( f"{pre_path}/{model_name}.h5ad" ) 
    
    
    
    
    
    
    


