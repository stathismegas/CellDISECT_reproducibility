#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/CellDISECT_reproducibility/figure_notebooks/kang_split_CD4_T

model_scripts="train_celldisect.py biolord_train.py scDisInFact_train.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-lotfollahi -n2 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=12000"
    done