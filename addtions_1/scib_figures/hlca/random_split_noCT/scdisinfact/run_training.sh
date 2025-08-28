#!/bin/bash

echo "module load cellgen/conda && conda activate disect && python HLCA_scdisinfact_train.py" | bsub -J train_hlca -G team361 -o logfile-train_hlca.out -e logfile-train_hlca.err -q gpu-lotfollahi -n1 -M256000 -R"select[mem>256000 && hname=='farm22-gpu0201'] rusage[mem=256000]" -gpu "mode=shared:num=1:gmem=8000"
