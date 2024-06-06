#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/kang/ablation_cfW

cfweights="0 0.005 0.01 0.03 0.05 0.08 0.1 0.3 0.5 0.8 1 5 10"
for d in $cfweights; do
    echo "python train_script.py ${d}" | \
        bsub -J ablation_cfW${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-lotfollahi -n2 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=10000"
    done