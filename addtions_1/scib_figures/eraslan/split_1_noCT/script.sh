#!/bin/bash

cd /nfs/users/nfs_a/aa34/mamad-works/hyperparameter_explore


echo "Running experiments with different latent dimensions..."
latent_dims="8 16 32 64 128"
for ld in $latent_dims; do
    echo "module load cellgen/conda && conda activate disect && python train_celldisect.py ${ld}" | \
        bsub -J ld_${ld}_split1 -G team361 -o logfile-ld_${ld}_split1.out -e logfile-ld_${ld}_split1.err -q gpu-lotfollahi -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done