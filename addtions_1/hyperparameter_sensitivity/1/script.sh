#!/bin/bash

cd /nfs/users/nfs_a/aa34/mamad-works

# First loop: iterate over learning rates with default n_layers
echo "Running experiments with different learning rates..."
lrs="0.0001 0.0003 0.0006 0.001 0.003 0.006 0.01 0.03 0.06 0.1"
for lr in $lrs; do
    echo "python train_celldisect.py ${lr}" | \
        bsub -J lr_${lr}_split4 -G team361 -o logfile-lr_${lr}_split4.out -e logfile-lr_${lr}_split4.err -q gpu-normal -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done

# Second loop: iterate over n_layers with default learning rate
echo "Running experiments with different n_layers..."
layers="1 2 4 6"
for nl in $layers; do
    echo "python train_celldisect.py 0.003 ${nl}" | \
        bsub -J nl_${nl}_split4 -G team361 -o logfile-nl_${nl}_split4.out -e logfile-nl_${nl}_split4.err -q gpu-normal -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done