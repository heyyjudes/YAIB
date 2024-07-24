#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

for hospital1 in "${hospital_ids_subset[@]:2}"; do
    echo "Training LSTM model for hospital ID: $hospital1"
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -m LSTM \
        -hp LSTMNet.hidden_dim=40 \
        -hi "$hospital1" \
        -hit "$hospital1" \
        --max_train=1500 \
        --complete-train
done