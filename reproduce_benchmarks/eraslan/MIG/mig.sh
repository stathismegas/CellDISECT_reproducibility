#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/eraslan/MIG

model_scripts="compute_mig_dis2p.py compute_mig_biolord.py compute_mig_scdisinfact.py"
split_key=("split_1" "split_2" "split_3" "split_4")

i=0
for split in "${split_key[@]}"; do
    i=$((i + 1))
    for d in $model_scripts; do
        echo "python ${d} \"${split}\"" | \
            bsub -J ${d}_${i} -G teichlab -o logfile-${d}_${i}.out -e logfile-${d}_${i}.err -q gpu-lotfollahi -n2 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=16000"
        done
    done