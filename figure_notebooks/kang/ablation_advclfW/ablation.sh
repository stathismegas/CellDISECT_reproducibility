#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/kang/ablation_advclfW

advclfweights="0 0.005 0.01 0.05 0.1 0.25 0.4 0.6 0.8 1 1.5 2 3 5 10"
split_key=("split_CD4 T" "split_CD14 Mono")

i = 0
for split in "${split_key[@]}"; do
    i = i + 1
    for d in $advclfweights; do
        echo "python train_script.py ${d} " \"${split}\"" | \
            bsub -J ablation_clfW${d}_${i} -G teichlab -o logfile-${d}_${i}.out -e logfile-${d}_${i}.err -q gpu-lotfollahi -n2 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=15000"
    done
done 