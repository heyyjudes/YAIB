#!/bin/bash

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
    -hi '264' \
    -hit '420' \
    --complete-train \
    --source-dir ../yaib_logs/eicu/Mortality24/LSTM/train264-test264/tuned/repetition_0/fold_0
