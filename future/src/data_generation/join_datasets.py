import os
import sys
import click
import pandas as pd
import numpy as np
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import preprocessing as pp

np.random.seed(1234)


@click.command()
@click.option('--input-dir', required=True, default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/bug_reports')
@click.option('--file-prefix', required=True, default='github_data_20')
@click.option('--embeddings', required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/future/embeddings_damevski/vectors_pad.txt')
@click.option('--output-file', required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/datasets/github/dataset.csv')
@click.option('--fraction',
              help='Fraction of original dataset to sample. If subset=1.0, the whole dataset is preserved.')
def join_files(**kwargs):
    dpath = kwargs['input_dir']
    prefix = kwargs['file_prefix']
    out_fpath = kwargs['output_file']
    w2v_model = read_w2v_model(kwargs['embeddings'])

    lines = list()
    header = None

    for root, dirs, files in os.walk(dpath):
        for file in files:
            if prefix in file and '.csv' in file:
                print('Processing {0}'.format(file))
                with open(os.path.join(root, file)) as f:
                    line = f.readline()
                    if header is None:
                        header = line
                    for line in f.readlines():
                        line = line.strip()

                        if len(line) > 0:
                            lines.append(line)

    print('Done')
    print('Save joined dataset to {0}'.format(out_fpath))
    out_dir = '/'.join(out_fpath.split('/')[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_fpath, 'w') as f:
        f.write(header)
        for line in lines:
            f.write(line + '\n')

    clear_data(out_fpath, w2v_model)

    if kwargs['fraction']:
        generate_subset(out_fpath, kwargs['fraction'])

    print('Done')


def clear_data(fpath, w2v_model):
    df = pd.read_csv(fpath)
    df = df.drop_duplicates(subset='issue_id')
    df = df[df['answer'].notna()]
    df = df[df['post'].notna()]
    df = df[df['question'].notna()]
    df = df[df['answer'].apply(lambda x: len(x.split()) > 3)]
    df = df.reset_index(drop=True)
    df = filter_by_w2v(df, w2v_model)
    df.to_csv(fpath, index=False)


def generate_subset(fpath, subset_ratio):
    df = pd.read_csv(fpath)
    df = df.sample(frac=subset_ratio).reset_index(drop=True)
    df.to_csv(fpath, index=False)


def read_w2v_model(path_in):
    path_out = '/'.join(path_in.split('/')[:-1]) + '/w2v_vectors.txt'
    glove2word2vec(path_in, path_out)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_out)
    return w2v_model


def filter_by_w2v(df, w2v_model):
    indexes_to_drop = list()

    for index, row in df.iterrows():
        if len(encode(row['post'], w2v_model)) == 0 or len(encode(row['question'], w2v_model)) == 0 or \
                len(encode(row['answer'], w2v_model)) == 0:
            indexes_to_drop.append(index)

    df.drop(df.index[indexes_to_drop], inplace=True)
    return df.reset_index(drop=True)


def encode(text, w2v_model):
    text = pp.clear_text(text, keep_punctuation=False)
    return [w2v_model.vocab[w].index for w in text.split() if w in w2v_model.vocab]


if __name__ == '__main__':
    join_files(sys.argv[1:])
