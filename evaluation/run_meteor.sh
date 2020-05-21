#!/bin/bash

PROJECT=/Users/ciborowskaa/VCU/Research/BugReportQA/

MODEL=GAN_question_generation
#MODEL=ranking_clarification_questions

DATASET=github_partial
CQ_DATA_DIR=$PROJECT/$MODEL/$DATASET
RESULTS_DIR=$PROJECT/$MODEL/evaluation/meteor/$SITENAME

# download from https://www.cs.cmu.edu/~alavie/METEOR/index.html#Download
METEOR=$PROJECT/meteor-1.5

#output file from prepare_data.py
TEST_SET=GAN_prediction_github_partial_seem_rao_epoch8.meteor
REFS = refs

java -Xmx2G -jar $METEOR/meteor-1.5.jar $CQ_DATA_DIR/$TEST_SET 	$CQ_DATA_DIR/$REFS \
										-l en -norm -r 1 \
									> $RESULTS_DIR/${TEST_SET}.meteor_results
