#!/bin/bash

icu-benchmarks \
    -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
    -n eicu \
    -t BinaryClassification \
    -tn Mortality24 \
    -hi "73-243" \
    -hit "73" \
    -m LogisticRegression \
    --addition_cap \
    --complete-train
