from scdisinfact_params import (
    reg_mmd_comm,
    reg_mmd_diff,
    reg_kl_comm,
    reg_kl_diff,
    reg_class,
    reg_gl, 
    Ks_shared,
    Ks_unshared,
    batch_size,
    nepochs, 
    interval,
    lr, 
    lambs
)

from scDisInFact import scdisinfact, create_scdisinfact_dataset
import scanpy as sc
import pandas as pd
import torch

adata = sc.read_h5ad("/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/HLCA_preprocessed_for_CellDISECT.h5ad" )
adata = adata[adata.layers['counts'].sum(1) != 0].copy()

adata.obs['one'] = pd.Categorical([1 for _ in range(adata.n_obs)]) # Dummy observation for batch (We treat Sample ID/Batch as covariate)

counts = adata.layers['counts']

meta_cells = adata.obs
condition_key = ['donor_id', 'sex', 'age_category', 'tissue']

Ks = [Ks_shared] + [Ks_unshared] * len(condition_key)

data_dict = create_scdisinfact_dataset(
    counts, meta_cells, condition_key=condition_key, batch_key="one")


# training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = scdisinfact(data_dict=data_dict, Ks=Ks, batch_size=batch_size, interval=interval, lr=lr,
                    reg_mmd_comm=reg_mmd_comm, reg_mmd_diff=reg_mmd_diff, reg_gl=reg_gl, reg_class=reg_class,
                    reg_kl_comm=reg_kl_comm, reg_kl_diff=reg_kl_diff, seed=0, device=device)
model.train()

losses = model.train_model(nepochs=nepochs, recon_loss="NB")
torch.save(model.state_dict(), '/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/training/scdisinfact/' +
           f"HLCA_scdisinfact_NoCT_defaultSettings_random_split.pth")