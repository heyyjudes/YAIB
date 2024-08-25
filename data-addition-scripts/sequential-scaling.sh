#!/bin/bash

train_values=("443-458" "443-458-167" "443-458-167-300" "443-73" "443-73-252" "443-73-252-338")

for train in "${train_values[@]}"; do
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$train"\
        -hit "443" \
        -hp LSTMNet.hidden_dim=40 \
        -m LSTM \
        --complete-train \
        --addition_cap=1200
done
