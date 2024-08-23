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

from ...parameters.biolord_params import train_method_params, n_latent, module_params, trainer_params

import biolord

import scanpy as sc

adata = sc.read_h5ad('/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/eraslan_preprocessed1212_split_deg.h5ad')
adata = adata[adata.layers['counts'].sum(1) != 0].copy()
cats = ['tissue', 'Sample ID', 'sex', 'Age_bin']
split_key = 'split_1'

biolord.Biolord.setup_anndata(
    adata=adata,
    ordered_attributes_keys=[],
    categorical_attributes_keys=cats,
    layer="counts",
)

model = biolord.Biolord(
    adata=adata,
    n_latent=n_latent,
    model_name=f"eraslan_biolord_NoCT_basicSettings_nb_{split_key}",
    module_params=module_params,
    train_classifiers=False,
    split_key=split_key,
    train_split='train',
    valid_split='val',
    test_split='test',)

model.train(
    max_epochs=train_method_params["max_epochs"],
    batch_size=train_method_params["batch_size"],
    plan_kwargs=trainer_params,
    early_stopping=train_method_params["early_stopping"],
    early_stopping_patience=train_method_params["early_stopping_patience"],
    check_val_every_n_epoch=train_method_params["check_val_every_n_epoch"],
    num_workers=train_method_params["num_workers"],
    enable_checkpointing=train_method_params["enable_checkpointing"],
)


model.save(f'/lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/models/biolord/eraslan_biolord_NoCT_earlierStop_basicSettings_nb_{split_key}')
