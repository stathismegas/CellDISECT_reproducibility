#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/kang

model_scripts="train_dis2p.py biolord_train.py scDisInFact_train.py"
split_key=("split_CD14 Mono" "split_CD4 T" "split_T" "split_CD8 T" "split_B" "split_DC" "split_CD16 Mono" "split_NK")

i=0
for split in "${split_key[@]}"; do
    i=$((i + 1))
    for d in $model_scripts; do
        echo "python ${d} \"${split}\"" | \
            bsub -J ${d}_${i} -G teichlab -o logfile-${d}_${i}.out -e logfile-${d}_${i}.err -q gpu-lotfollahi -n1 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=12000"
        done
    done