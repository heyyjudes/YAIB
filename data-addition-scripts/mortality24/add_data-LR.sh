#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

for hospital1 in "${hospital_ids_subset[@]}"; do
    for hospital2 in "167" "252"; do
        echo "Training model for hospital ID: $hospital1-$hospital2 testing for hospital $hospital2"
        if [ "$hospital1" != "$hospital2" ]; then
            NVIDIA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 icu-benchmarks \
                -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
                -n eicu \
                -t BinaryClassification \
                -tn Mortality24 \
                -hi "$hospital1-$hospital2" \
                -hit "$hospital2" \
                --complete-train \
                -m LogisticRegression \
                --addition_cap 1000 
        else
            NVIDIA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 icu-benchmarks \
                -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
                -n eicu \
                -t BinaryClassification \
                -tn Mortality24 \
                -hi "$hospital1" \
                -hit "$hospital1" \
                --complete-train \
                -m LogisticRegression \
                --max_train 2000
        fi
    done
done