#!/bin/bash

DATASET=github_20K
RESULT_FILE=evpi_1
REFS_NUMBER=1

PROJECT=/Users/ciborowskaa/VCU/Research/BugReportQA
RESULTS_DIR=$PROJECT/results
DATA_FPATH=$RESULTS_DIR/$DATASET/${RESULT_FILE}.csv
OUTPUT_DIR=$RESULTS_DIR/$DATASET/output

# download from https://www.cs.cmu.edu/~alavie/METEOR/index.html#Download
METEOR=$PROJECT/meteor-1.5/meteor-1.5

python evaluation.py --results-fpath $DATA_FPATH \
                     --output-dir $OUTPUT_DIR \
                     --ref-no $REFS_NUMBER \
                     --meteor $METEOR

