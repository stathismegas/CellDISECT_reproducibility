#!/bin/bash

cd /lustre/scratch126/cellgen/team205/aa34/Arian/Dis2P/dis2p_reproducibility/reproduce_benchmarks/haber

model_scripts="train_dis2p.py biolord_train.py scDisInFact_train.py"

split_key=(
"split_allOut_Enterocyte.Progenitor_salmonella"
"split_allOut_Stem_salmonella"
"split_allOut_TA.Early_salmonella"
"split_allOut_TA_salmonella"
"split_allOut_Tuft_salmonella"
"split_allOut_Enterocyte_salmonella"
"split_allOut_Goblet_salmonella"
"split_allOut_Endocrine_salmonella"
"split_allOut_Enterocyte.Progenitor_hpoly10"
"split_allOut_Stem_hpoly10"
"split_allOut_TA.Early_hpoly10"
"split_allOut_TA_hpoly10"
"split_allOut_Tuft_hpoly10"
"split_allOut_Enterocyte_hpoly10"
"split_allOut_Goblet_hpoly10"
"split_allOut_Endocrine_hpoly10"
"split_targetOut_Enterocyte.Progenitor_salmonella"
"split_targetOut_Stem_salmonella"
"split_targetOut_TA.Early_salmonella"
"split_targetOut_TA_salmonella"
"split_targetOut_Tuft_salmonella"
"split_targetOut_Enterocyte_salmonella"
"split_targetOut_Goblet_salmonella"
"split_targetOut_Endocrine_salmonella"
"split_targetOut_Enterocyte.Progenitor_hpoly10"
"split_targetOut_Stem_hpoly10"
"split_targetOut_TA.Early_hpoly10"
"split_targetOut_TA_hpoly10"
"split_targetOut_Tuft_hpoly10"
"split_targetOut_Enterocyte_hpoly10"
"split_targetOut_Goblet_hpoly10"
"split_targetOut_Endocrine_hpoly10"
)


i=0
for split in "${split_key[@]}"; do
    i=$((i + 1))
    for d in $model_scripts; do
        echo "python ${d} \"${split}\"" | \
            bsub -J ${d}_${i} -G teichlab -o logfile-${d}_${i}.out -e logfile-${d}_${i}.err -q gpu-lotfollahi -n1 -M16000 -R"select[mem>16000] rusage[mem=16000]" -gpu "mode=shared:num=1:gmem=12000"
        done
    done