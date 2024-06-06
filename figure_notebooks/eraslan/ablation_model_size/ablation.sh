#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/ablation_model_size

n_layers="2 4"
n_hidden="128 512 1024 2048"
n_latent="10 128 512 1024"
for layer in $n_layers; do
    for hidden in $n_hidden; do
        for latent in $n_latent; do
            echo "python train_script.py ${layer} ${hidden} ${latent}" | \
                bsub -J ablation_model_size${layer}_${hidden}_${latent} -G teichlab -o logfile-${layer}_${hidden}_${latent}.out -e logfile-${layer}_${hidden}_${latent}.err -q gpu-normal -n2 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=10000"
            done
        done
    done