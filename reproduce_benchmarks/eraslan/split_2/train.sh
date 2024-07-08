#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/eraslan/split_2

model_scripts="train_dis2p.py biolord_train.py scDisInFact_train.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d}_split2 -G teichlab -o logfile-${d}_split2.out -e logfile-${d}_split2.err -q gpu-lotfollahi -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=40000"
    done