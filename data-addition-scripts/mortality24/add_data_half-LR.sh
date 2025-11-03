#!/bin/bash

# experiment output format will be f"train{args.hospital_id}-test{args.hospital_id_test}-n{args.addition_cap}"

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

for hospital1 in "${hospital_ids[@]}"; do
    for hospital2 in "${hospital_ids[@]}"; do
        echo "Training model for hospital ID: $hospital1-$hospital2 testing for hospital $hospital2"
        icu-benchmarks \
            -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
            -n eicu \
            -t BinaryClassification \
            -tn Mortality24 \
            -hi "$hospital1-$hospital2" \
            -hit "$hospital2" \
            --complete-train \
            -m LogisticRegression \
            --addition_cap=1000 
    done
done


