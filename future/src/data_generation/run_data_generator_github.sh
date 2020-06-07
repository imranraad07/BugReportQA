#!/bin/bash

DATA_DIR=data
SITE_NAME=github_20K

SCRIPTS_DIR=src/data_generation
LUCENE_DIR=lucene

mkdir -p $DATA_DIR/$SITE_NAME

rm -r $DATA_DIR/$SITE_NAME/post_docs
rm -r $DATA_DIR/$SITE_NAME/post_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

rm -r $DATA_DIR/$SITE_NAME/ques_docs
rm -r $DATA_DIR/$SITE_NAME/ques_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/ques_docs

python $SCRIPTS_DIR/data_generator.py   --lucene_dir $LUCENE_DIR \
                                        --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs \
                                        --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts.txt \
                                        --post_data_tsv $DATA_DIR/$SITE_NAME/post_data.tsv \
                                        --qa_data_tsv $DATA_DIR/$SITE_NAME/qa_data.tsv \
                                        --utility_data_tsv $DATA_DIR/$SITE_NAME/utility_data.tsv \
                                        --github_csv $DATA_DIR/$SITE_NAME/dataset.csv
