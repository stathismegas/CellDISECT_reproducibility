n_latent = 32

module_params = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "nb",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "seed": 42,
}

trainer_params = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-4,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}

train_method_params = {
    "max_epochs": 500,
    "batch_size": 512,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "check_val_every_n_epoch": 5,
    "num_workers": 1,
    "enable_checkpointing": False,
}

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

