#!/bin/bash

train_values=("199-73-252" "199-73-252-443" "199-300-458" "199-300-458-167")

for train in "${train_values[@]}"; do
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$train"\
        -hit "199" \
        -hp LSTMNet.hidden_dim=40 \
        -m LSTM \
        --complete-train \
        --addition_cap
done
