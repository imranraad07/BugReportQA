#!/bin/bash

DATA_DIR=../future/data

DATASET=dummy1K
RESULTS=../results/$DATASET/random_baseline.csv

DEVICE=cuda

python random_baseline.py \
        --post-tsv $DATA_DIR/$DATASET/post_data.tsv \
        --test-ids $DATA_DIR/$DATASET/test_ids.txt \
        --qa-tsv $DATA_DIR/$DATASET/qa_data.tsv \
        --output-ranking-file $RESULTS \
