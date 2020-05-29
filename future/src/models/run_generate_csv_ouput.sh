#!/bin/bash

DATA_DIR=data
SITE_NAME=github

SCRIPTS_DIR=src/models

python $SCRIPTS_DIR/generate_csv_output.py	--data_dir $DATA_DIR/$SITE_NAME \
								                	--test_ids $DATA_DIR/$SITE_NAME/test_ids \
                                  --qa_data_tsv $DATA_DIR/$SITE_NAME/qa_data.tsv \
                                  --github_csv $DATA_DIR/$SITE_NAME/github_data.csv \
                                  --epoch0 $DATA_DIR/$SITE_NAME/test_predictions_evpi.out.epoch0 \
                                  --epoch13 $DATA_DIR/$SITE_NAME/test_predictions_evpi.out.epoch13 \
                                  --epoch19 $DATA_DIR/$SITE_NAME/test_predictions_evpi.out.epoch19 \
                                  --test_predictions_epoch0_csv $DATA_DIR/$SITE_NAME/test_predictions_epoch0.csv \
                                  --test_predictions_epoch13_csv $DATA_DIR/$SITE_NAME/test_predictions_epoch13.csv \
                                  --test_predictions_epoch19_csv $DATA_DIR/$SITE_NAME/test_predictions_epoch19.csv