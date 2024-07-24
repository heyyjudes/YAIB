#!/bin/bash
hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")
for hospital1 in "${hospital_ids_subset[@]}"; do
    for hospital2 in "${hospital_ids_subset[@]}"; do
        if [ "$hospital1" != "$hospital2" ]; then
            echo "Eval model from hospital ID: $hospital1 on hospital $hospital2"
            icu-benchmarks \
                --eval \
                -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
                -n eicu \
                -t BinaryClassification \
                -tn Mortality24 \
                -m LSTM \
                --generate_cache \
                --load_cache \
                -s 2222 \
                -l ../yaib_logs \
                -sn eicu \
                -hi "$hospital1" \
                -hit "$hospital2" \
                --complete-train \
                --max_train=1500 \
                --source-dir "../yaib_logs/eicu/Mortality24/LSTM/train-test$hospital1-n1500/small/repetition_0/fold_0"
        fi
    done 
done 