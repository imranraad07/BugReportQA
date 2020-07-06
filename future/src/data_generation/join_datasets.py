import csv
import os
import sys

import click
import gensim
import numpy as np
import pandas as pd
import preprocessing as pp
from gensim.scripts.glove2word2vec import glove2word2vec

from future.src.data_generation.github_text_filter import *

np.random.seed(1234)


@click.command()
@click.option('--input-issue-titles-dir', required=True, default='../../../data/bug_reports/github_issue_titles.csv')
@click.option('--input-dir', required=True, default='../../../data/bug_reports')
@click.option('--file-prefix', required=True, default='github_data_20')
@click.option('--embeddings', required=True,
              default='../../embeddings_damevski/vectors_pad.txt')
@click.option('--output-file', required=True,
              default='../../../data/datasets/github/dataset.csv')
@click.option('--fraction',
              help='Fraction of original dataset to sample. If subset=1.0, the whole dataset is preserved.')
def join_files(**kwargs):
    dpath = kwargs['input_dir']
    prefix = kwargs['file_prefix']
    out_fpath = kwargs['output_file']
    w2v_model = read_w2v_model(kwargs['embeddings'])

    issue_title_path = kwargs['input_issue_titles_dir']
    print(issue_title_path)
    issue_titles = {}
    with open(issue_title_path) as csv_data_file:
        csv_reader = csv.reader((line.replace('\0', '') for line in csv_data_file))
        next(csv_reader)
        for row in csv_reader:
            issue_titles[row[0]] = row[1]

    header = None
    br_count = 0
    filtered_br = 0
    br_reports = []
    issue_ids = []
    for root, dirs, files in os.walk(dpath):
        for file in files:
            if prefix in file and '.csv' in file:
                if '_edit.csv' in file:
                    print('Processing {0}'.format(file))
                    with open(os.path.join(root, file)) as f:
                        csv_reader = csv.reader((line.replace('\0', '') for line in f))
                        if header is None:
                            header = next(csv_reader)
                        else:
                            next(csv_reader)
                        for row in csv_reader:
                            if row[2] in issue_ids:
                                continue
                            repo = row[0]
                            issue_link = row[1]
                            issue_id = row[2]
                            post = row[3]
                            question = row[4]
                            answer = row[5]
                            if issue_id not in issue_titles or should_title_be_filtered(issue_titles[issue_id]) is True:
                                filtered_br = filtered_br + 1
                                continue
                            if should_post_be_filtered(post) is True:
                                filtered_br = filtered_br + 1
                                continue
                            if should_question_be_filtered(question) is True:
                                filtered_br = filtered_br + 1
                                continue
                            post = filter_nontext(post)
                            question = filter_nontext(question)
                            answer = filter_nontext(answer)
                            br_reports.append([repo, issue_link, issue_id, post, question, answer])
                            br_count = br_count + 1
                            issue_ids.append(issue_id)

    for root, dirs, files in os.walk(dpath):
        for file in files:
            if prefix in file and '.csv' in file:
                if '_edit.csv' in file:
                    continue
                print('Processing {0}'.format(file))
                with open(os.path.join(root, file)) as f:
                    csv_reader = csv.reader((line.replace('\0', '') for line in f))
                    if header is None:
                        header = next(csv_reader)
                    else:
                        next(csv_reader)
                    for row in csv_reader:
                        if row[2] in issue_ids:
                            continue
                        repo = row[0]
                        issue_link = row[1]
                        issue_id = row[2]
                        post = row[3]
                        question = row[4]
                        answer = row[5]
                        if issue_id not in issue_titles or should_title_be_filtered(issue_titles[issue_id]) is True:
                            filtered_br = filtered_br + 1
                            continue
                        if should_post_be_filtered(post) is True:
                            filtered_br = filtered_br + 1
                            continue
                        if should_question_be_filtered(question) is True:
                            filtered_br = filtered_br + 1
                            continue
                        post = filter_nontext(post)
                        question = filter_nontext(question)
                        answer = filter_nontext(answer)
                        br_reports.append([repo, issue_link, issue_id, post, question, answer])
                        br_count = br_count + 1
                        issue_ids.append(issue_id)

    print("total_bug_reports:", br_count)
    print("filtered_bug_reports:", filtered_br)
    print('Done')
    print('Save joined dataset to {0}'.format(out_fpath))
    out_dir = '/'.join(out_fpath.split('/')[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    csv_file = open(out_fpath, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    for br in br_reports:
        csv_writer.writerow(br)
    csv_file.close()

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
    csv.field_size_limit(sys.maxsize)
    join_files(sys.argv[1:])
