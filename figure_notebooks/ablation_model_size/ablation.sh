#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/ablation_cfW

n_layers="1 2 3 4 5"
n_hidden="32 64 128 256 512 1024 2048"
n_latent="10 32 64 128 256 512 1024"
for layer in $n_layers; do
    for hidden in $n_hidden; do
        for latent in $n_latent; do
            echo "python train_script.py ${layer} ${hidden} ${latent}" | \
                bsub -J ablation_model_size${layer}_${hidden}_${latent} -G teichlab -o logfile-${layer}_${hidden}_${latent}.out -e logfile-${layer}_${hidden}_${latent}.err -q gpu-normal -n2 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
            done
        done
    done