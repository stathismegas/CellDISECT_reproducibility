#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/figure_notebooks/eraslan/MIG

model_scripts="compute_mig_celldisect.py compute_mig_biolord.py compute_mig_scdisinfact.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d} -G teichlab -o logfile-${d}.out -e logfile-${d}.err -q gpu-lotfollahi -n2 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=16000"
    done