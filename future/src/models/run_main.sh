#!/bin/bash

SCRIPTS_DIR=src/models/
DATA_DIR=data
EMB_DIR=embeddings_damevski

DATASET=github_partial_2008-2013_part1_small
RESULTS=$DATA_DIR/$DATASET/ranking.csv

DEVICE=cuda

python $SCRIPTS_DIR/main.py \
        --embeddings $EMB_DIR/vectors_pad.txt \
        --post-tsv $DATA_DIR/$DATASET/post_data.tsv \
        --train-ids $DATA_DIR/$DATASET/train_ids.txt \
        --test-ids $DATA_DIR/$DATASET/test_ids.txt \
        --qa-tsv $DATA_DIR/$DATASET/qa_data.tsv \
        --utility-tsv $DATA_DIR/$DATASET/utilit_data.tsv \
        --output-ranking-file $RESULTS \
        --device $DEVICE \
        --batch-size 10 \
        --n-epochs 30 \
        --max-p-len 300 \
        --max-q-len 100 \
        --max-a-len 100
