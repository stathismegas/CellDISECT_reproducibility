#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/antony

names="optimal no_clf low_clf"

i=0
for d in $names; do
    echo "python antony_process.py ${d}" | \
        bsub -J antonyProccess${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-lotfollahi -n2 -M128000 -R"select[mem>128000] rusage[mem=128000]" -gpu "mode=shared:num=1:gmem=70000"
done
