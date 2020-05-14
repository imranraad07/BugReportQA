#!/bin/bash

DATA_FPATH=/Users/ciborowskaa/VCU/Research/BugReportQA/data/bug_reports/github_data.csv
OUT_DATA_FPATH=/Users/ciborowskaa/VCU/Research/BugReportQA/GAN_question_generation/embeddings/github/github.data.txt
SCRIPT_DIR=/Users/ciborowskaa/VCU/Research/BugReportQA/GAN_question_generation/src/embedding_generation

python $SCRIPT_DIR/extract_github_data.py	--data-csv $DATA_FPATH \
									--output-fpath OUT_DATA_FPATH \
									--triplets \


