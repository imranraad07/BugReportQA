#!/bin/bash

DATA_DIR=data/datasets
SITE_NAME=datasets_final_tag

SCRIPTS_DIR=future/src/data_generation
LUCENE_DIR=future/lucene

mkdir -p $DATA_DIR/$SITE_NAME

rm -r $DATA_DIR/$SITE_NAME/post_docs
rm -r $DATA_DIR/$SITE_NAME/post_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/post_docs

rm -r $DATA_DIR/$SITE_NAME/ques_docs
rm -r $DATA_DIR/$SITE_NAME/ques_doc_indices
mkdir -p $DATA_DIR/$SITE_NAME/ques_docs

python $SCRIPTS_DIR/data_generator_question.py --lucene_dir $LUCENE_DIR \
  --lucene_docs_dir $DATA_DIR/$SITE_NAME/post_docs \
  --lucene_similar_posts $DATA_DIR/$SITE_NAME/lucene_similar_posts_qa.txt \
  --qa_data_tsv $DATA_DIR/$SITE_NAME/data_avg_qa.tsv \
  --github_csv $DATA_DIR/$SITE_NAME/dataset.csv