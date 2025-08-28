#!/bin/bash
cd /nfs/users/nfs_a/aa34/mamad-works

# First loop: iterate over classification weights
echo "Running experiments with different classification weights..."
clf_weights="0.05 0.1 0.5 1.0"
for clf_w in $clf_weights; do
    echo "python train_celldisect.py ${clf_w}" | \
        bsub -J clf_${clf_w}_split4 -G team361 -o logfile-clf_${clf_w}_split4.out -e logfile-clf_${clf_w}_split4.err -q gpu-normal -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done

# Second loop: iterate over adversarial weights
echo "Running experiments with different adversarial weights..."
adv_weights="0.01 0.1 0.14 0.5"
for adv_w in $adv_weights; do
    echo "python train_celldisect.py default ${adv_w}" | \
        bsub -J adv_${adv_w}_split4 -G team361 -o logfile-adv_${adv_w}_split4.out -e logfile-adv_${adv_w}_split4.err -q gpu-normal -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done

# Third loop: iterate over counterfactual weights
echo "Running experiments with different counterfactual weights..."
cf_weights="0.05 0.1 0.5 0.8 1.0"
for cf_w in $cf_weights; do
    echo "python train_celldisect.py default default ${cf_w}" | \
        bsub -J cf_${cf_w}_split4 -G team361 -o logfile-cf_${cf_w}_split4.out -e logfile-cf_${cf_w}_split4.err -q gpu-normal -n1 -M32000 -R"select[mem>32000] rusage[mem=32000]" -gpu "mode=shared:num=1:gmem=20000"
done