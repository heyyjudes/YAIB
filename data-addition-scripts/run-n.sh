#!/bin/bash

max_train_values=(400 800 1000 1200 1500 2000)

for max_train in "${max_train_values[@]}"; do
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "443" \
        -hit "443" \
        -m LogisticRegression \
        --complete-train \
        --max_train="$max_train"
done
