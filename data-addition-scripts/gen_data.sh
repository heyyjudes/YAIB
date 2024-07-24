#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

for hospital1 in "${hospital_ids_subset[@]}"; do
    echo "Training model for hospital ID: $hospital1 testing for hospital $hospital1"
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$hospital1" \
        -hit "$hospital1" \
        --complete-train \
        -m LogisticRegression \
        --save_data
    #done
done