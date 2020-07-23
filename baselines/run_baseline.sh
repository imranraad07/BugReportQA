#!/bin/bash

DATA_DIR=../data/datasets

DATASET=datasets_final_tag
RESULTS=../results/$DATASET/ranking_baseline.csv

DEVICE=cuda

python baseline_utility.py \
        --post-tsv $DATA_DIR/$DATASET/post_data.tsv \
        --test-ids $DATA_DIR/$DATASET/test_ids.txt \
        --qa-tsv $DATA_DIR/$DATASET/qa_data.tsv \
        --output-ranking-file $RESULTS \
