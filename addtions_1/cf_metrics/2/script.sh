#!/bin/bash

echo "module load cellgen/conda && conda activate disect && python benchmark_split_4_noCT_2.py" | bsub -J benchmark_4_noCT -G team361 -o logfile-benchmark_4_noCT_round2.out -e logfile-benchmark_4_noCT_round2.err -q gpu-lotfollahi -n1 -M32000 -R"select[mem>32000 && hname=='farm22-gpu0201'] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=40000"