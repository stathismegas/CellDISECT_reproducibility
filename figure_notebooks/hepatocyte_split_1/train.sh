#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/hepatocyte_split_1

model_scripts="train_dis2p.py biolord_train.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-normal -n2 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=12000"
    done