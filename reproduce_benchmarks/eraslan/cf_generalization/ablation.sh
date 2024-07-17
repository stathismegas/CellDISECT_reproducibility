#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility_clean/reproduce_benchmarks/eraslan/cf_generalization

recon_weight="0 1 5 10 15 20"
cf_weight="0 0.5 1 1.5 3 5"
for recon in $recon_weight; do
    for cf in $cf_weight; do
        echo "python train_dis2p.py ${recon} ${cf}" | \
            bsub -J ablation_bingo_recon_cf_${recon}_${cf} -G teichlab -o logfile-${recon}_${cf}.out -e logfile-${recon}_${cf}.err -q gpu-lotfollahi -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=40000"
    done
done