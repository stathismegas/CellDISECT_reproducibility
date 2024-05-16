from scDisInFact import scdisinfact, create_scdisinfact_dataset
import scanpy as sc
import pandas as pd
import torch
split_key = 'split_1'
adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')

adata = adata[adata.obs[split_key] == 'train'].copy()
adata.obs['one'] = pd.Categorical([1 for _ in range(adata.n_obs)]) # Dummy observation for batch (We treat Sample ID/Batch as covariate)

counts = adata.layers['counts']

meta_cells = adata.obs
condition_key = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']



data_dict = create_scdisinfact_dataset(counts, meta_cells, condition_key = condition_key, batch_key = "one")


# declare latent dimensions, we have two condition types, so there are three element corresponding to 
# shared-bio factor, unshared-bio factor for condition 1, unshared-bio factor for condition 2
# default setting of hyper-parameters
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [40, 40, 40, 40, 40, 40]

batch_size = 64
nepochs = 300
interval = 10
lr = 5e-4
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]


# training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                    reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()

losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
torch.save(model.state_dict(), '/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/scDisInfact/' + f"model_40latent_300ep_CoarseCT_{split_key}_{Ks}_{lambs}_{batch_size}_{nepochs}_{lr}.pth")
