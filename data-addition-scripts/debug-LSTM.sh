#!/bin/bash
echo "Training model for hospital ID: $hospital1-$hospital2 testing for hospital $hospital2"
icu-benchmarks \
    -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
    -n eicu \
    -t BinaryClassification \
    -tn Mortality24 \
    -hi "338-167" \
    -hit 167 \
    --complete-train \
    -hp LSTMNet.hidden_dim=40 \
    -m LSTM \
    --addition_cap