#!/bin/bash

max_train_values=(2000 3000 4000)
hospital_ids=(443)

for hospital1 in "${hospital_ids[@]}"; do
    for max_train in "${max_train_values[@]}"; do
        icu-benchmarks \
            -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
            -n eicu \
            -t BinaryClassification \
            -tn Mortality24 \
            -hi "443-74-264-420-243-338-199-458-300-188-252-167" \
            -hit "$hospital1" \
            -m LogisticRegression \
            --complete-train \
            --max_train="$max_train" \
            --addition-cap
    done
done
