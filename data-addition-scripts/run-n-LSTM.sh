#!/bin/bash

max_train_values=(2500)

for max_train in "${max_train_values[@]}"; do
    icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "443" \
        -hit "443" \
        -hp LSTMNet.hidden_dim=40 \
        -m LSTM \
        --complete-train \
        --max_train="$max_train"
done