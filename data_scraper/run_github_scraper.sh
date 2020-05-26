#!/bin/bash

DATA_DIR=data/repos
OUTPUT_DIR=data/bug_reports
SCRIPTS_DIR=data_scraper

TYPE=edit
#TYPE=parse

python $SCRIPTS_DIR/github_scraper.py \
  --type $TYPE \
  --repo_csv $DATA_DIR/repos_final2008.csv \
  --output_csv $OUTPUT_DIR/github_data_2008_edit.csv
