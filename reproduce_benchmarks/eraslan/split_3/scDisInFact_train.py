import sys, importlib
from pathlib import Path

file = Path(__file__).resolve()
level=3
parent, top = file.parent, file.parents[level]
sys.path.append(str(top))
try:
    sys.path.remove(str(parent))
except ValueError: # already removed
    pass
__package__ = '.'.join(parent.parts[len(top.parts):])
importlib.import_module(__package__)

from ...parameters.scdisinfact_params import (
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
split_key = 'split_3'
adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()

adata = adata[adata.obs[split_key] == 'train'].copy()
adata.obs['one'] = pd.Categorical([1 for _ in range(adata.n_obs)]) # Dummy observation for batch (We treat Sample ID/Batch as covariate)

counts = adata.layers['counts']

meta_cells = adata.obs
condition_key = ['tissue', 'Sample ID', 'Age_bin', 'CoarseCellType']

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
torch.save(model.state_dict(), '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/scDisInfact/' +
           f"eraslan_scdisinfact_defaultSettings_f{split_key}.pth")
