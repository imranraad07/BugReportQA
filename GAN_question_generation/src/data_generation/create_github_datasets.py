import os
import sys
import click
import pandas as pd
import numpy as np

np.random.seed(1234)


@click.command()
@click.option('--in-directory', help='Input directory containing dataset.csv, trainig/tuning/testing ids',
              required=True)
@click.option('--out-directory', help='Directory to save output dataset files', required=True)
def run(*args, **kwargs):
    in_dir = kwargs['in_directory']
    out_dir = kwargs['out_directory']

    dataset = pd.read_csv(os.path.join(in_dir, 'dataset.csv'))

    prepare_dataset(dataset, in_dir, out_dir, 'train')
    prepare_dataset(dataset, in_dir, out_dir, 'tune')
    prepare_dataset(dataset, in_dir, out_dir, 'test')


def prepare_dataset(dataset, in_dir, out_dir, type):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(in_dir, type + '_ids.txt')) as f:
        ids = set([x.strip() for x in f.readlines()])

    with open(os.path.join(out_dir, type + '_context.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            if row['issue_id'] in ids:
                br = preprocess(row['post'])
                f.write(br + '\n')

    with open(os.path.join(out_dir, type + '_answer.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            if row['issue_id'] in ids:
                answer = preprocess(row['answer'])
                f.write(answer + '\n')

    with open(os.path.join(out_dir, type + '_question.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            if row['issue_id'] in ids:
                question = preprocess(row['question'])
                f.write(question + '\n')

    with open(os.path.join(out_dir, type + '_ids.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            if row['issue_id'] in ids:
                id = row['issue_id']
                f.write(id + '\n')


def preprocess(text):
    if len(text) < 10:
        print('Suspiciously short text {0}'.format(text))
    text = text.lower().strip().replace('\r\n', ' ').replace('\n', ' ')
    text_filtered = ''
    for c in text:
        if c in '!@$%^&*()[]{};:,./<>?\|`~-=':
            text_filtered += ' '
        else:
            text_filtered += c
    return text_filtered


def preprocess_id(id):
    tokens = id.split('/')
    repo = tokens[-3]
    no = tokens[-1]
    return repo + '_' + no


if __name__ == '__main__':
    run(sys.argv[1:])
