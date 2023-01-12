#!/bin/bash

eval names=$4
eval source_dirs=$6
for var in ${names[@]}
do
    for i in ${!names[@]}
    do
        icu-benchmarks evaluate \
            -e $1 \
            -l $2 \
            -d $3$var \
            -n $var \
            -sn ${names[$i]} \
            --source-dir $5${names[$i]}/${source_dirs[$i]} \
            -c \
            -s 1111
    done
done
