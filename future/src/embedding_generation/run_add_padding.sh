#!/bin/bash

SCRIPTS_DIR=future/src/embedding_generation
EMB_DIR=future/embeddings_damevski/

python $SCRIPTS_DIR/add_padding_token.py \
                  --glove-vectors $EMB_DIR/vectors.txt \
                  --output-vectors $EMB_DIR/vectors_pad.txt

