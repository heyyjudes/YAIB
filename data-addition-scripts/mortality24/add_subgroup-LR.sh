#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

hospital_ids_subset=("${hospital_ids[@]}")
subgroups=("white" "black" "asian" "other")

for hospital1 in "${hospital_ids_subset[@]}"; do
    for hospital2 in "${hospital_ids_subset[@]}"; do
        for subgroup in  "${subgroups[@]}" ; do
            # if [ "$hospital1" != "$hospital2" ]; then
            echo "Training model for hospital ID: $hospital1-$hospital2 testing for hospital $hospital2"
            NVIDIA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 icu-benchmarks \
                -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
                -n eicu \
                -t BinaryClassification \
                -tn Mortality24 \
                -hi "$hospital1-$hospital2" \
                -hit "$hospital2" \
                --complete-train \
                -m LogisticRegression \
                -adds "$subgroup"  \
            # fi 
        done
    done
done

