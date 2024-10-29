#!/bin/bash


icu-benchmarks \
        -d /home/ubuntu/projects/more-data-more-problems/YAIB/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "73-264-420"\
        -hit "188-252-167" \
        -m LGBMClassifier \
        -hp LGBMClassifier.min_child_samples=10 \
        --complete-train