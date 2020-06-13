#!/bin/bash

PROJECT=/Users/ciborowskaa/VCU/Research/BugReportQA/

MODEL=future

DATASET=github_20K
DATA_DIR=$PROJECT/results/$DATASET

# download from https://www.cs.cmu.edu/~alavie/METEOR/index.html#Download
METEOR=$PROJECT/meteor-1.5

#output file from prepare_data.py
TEST_SET=evpi_256_questions.txt
REFS=evpi_256_refs.txt

java -Xmx2G -jar $METEOR/meteor-1.5.jar $DATA_DIR/$TEST_SET 	$DATA_DIR/$REFS \
										-l en -norm -r 1 \
									> $DATA_DIR/${TEST_SET}.meteor_results
