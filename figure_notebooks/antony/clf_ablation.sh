#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/antony

clf_weight="0.20 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038 0.40"
for clf in $clf_weight; do
    echo "python cfl_ablation.py ${clf}" | \
        bsub -J clf_ablation${clf} -G teichlab -o logfile-clf_ablation${clf}.out -e logfile-clf_ablation${clf}.err -q gpu-lotfollahi -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=40000"
done