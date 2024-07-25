#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/haber

model_scripts="train_dis2p.py biolord_train.py scDisInFact_train.py"
for d in $model_scripts; do
    echo "python ${d}" | \
        bsub -J ${d}_split1 -G teichlab -o logfile-${d}_split1.out -e logfile-${d}_split1.err -q gpu-lotfollahi -n1 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=16000"
    done