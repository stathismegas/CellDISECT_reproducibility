#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/eraslan/split_2

model_scripts="train_dis2p.py biolord_train.py scDisInFact_train.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-lotfollahi -n2 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=40000"
    done