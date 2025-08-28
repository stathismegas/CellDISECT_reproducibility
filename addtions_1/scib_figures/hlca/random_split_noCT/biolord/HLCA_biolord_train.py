from biolord_params import train_method_params, n_latent, module_params, trainer_params

import biolord

import scanpy as sc

adata = sc.read_h5ad("/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/HLCA_preprocessed_for_CellDISECT.h5ad" )
adata = adata[adata.layers['counts'].sum(1) != 0].copy()

cats = ['donor_id', 'sex', 'age_category', 'tissue']

biolord.Biolord.setup_anndata(
    adata=adata,
    ordered_attributes_keys=[],
    categorical_attributes_keys=cats,
    layer="counts",
)

model = biolord.Biolord(
    adata=adata,
    n_latent=n_latent,
    model_name=f"HLCA_biolord_NoCT_basicSettings_nb_random_split",
    module_params=module_params,
    train_classifiers=False)

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


model.save(f'/lustre/scratch126/cellgen/lotfollahi/aa34/mamad-works/HLCA/training/biolord/HLCA_biolord_NoCT_earlierStop_basicSettings_nb_random_split')