#!/bin/bash

icu-benchmarks \
    -d /home/ubuntu/projects/more-data-more-problems/YAIB-cohorts/data/mortality24/eicu \
    -n eicu \
    -t BinaryClassification \
    -tn Mortality24 \
    -hi "264" \
    -hit "73" \
    -m LGBMClassifier \
    -hp LGBMClassifier.min_child_samples=10 \
    --complete-train