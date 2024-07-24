#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

for ((i=0; i<${#hospital_ids_subset[@]}; i++)); do
    hospital1=${hospital_ids_subset[i]}
    for ((j=i+1; j<${#hospital_ids_subset[@]}; j++)); do
        hospital2=${hospital_ids_subset[j]}
        echo "Training and testing on hospitals IDs: $hospital1-$hospital2"
        icu-benchmarks \
            -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
            -n eicu \
            -t BinaryClassification \
            -tn Mortality24 \
            -hi "$hospital1-$hospital2" \
        -hp LSTMNet.hidden_dim=40 \
        -m LSTM
    done
done