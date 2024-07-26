arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.2,
 'weighted_classifier': True,
}
train_dict = {
 'max_epochs': 100,
 'batch_size': 256,
 'recon_weight': 20,
 'cf_weight': 1.5,
 'beta': 0.003,
 'clf_weight': 0.8,
 'adv_clf_weight': 0.015,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 15,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
 'n_epochs_pretrain_ae': 10,
}

plan_kwargs = {
 'lr': 0.01,
 'weight_decay': 0.0005,
 'new_cf_method': True,
 'lr_patience': 6,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}

