import os
import sys
import click
import pandas as pd
import numpy as np

np.random.seed(1234)


@click.command()
@click.option('--in-dataset',
              help='Input dataset file',
              default='/Users/ciborowskaa/VCU/Research/GAN/')
@click.option('--out-directory',
              help='Directory to save output dataset files',
              required=True)
@click.option('--ratios',
              help='Ratio of training/tuning/testing dataset. Format: [train_ratio tune_ratio test_ratio]',
              type=(float, float, float),
              default=[0.6, 0.2, 0.2],
              required=True)
def run(*args, **kwargs):
    in_dataset = kwargs['in_dataset']
    out_dir = kwargs['out_directory']
    train_ratio, tune_ratio, test_ratio = kwargs['ratios']
    if train_ratio + tune_ratio + test_ratio != 1:
        raise ValueError('Ratios of training + tuning + testing datasets should sum up to 1.')

    dataset = pd.read_csv(in_dataset, names=['repo_id', 'issue_id', 'br', 'question', 'answer'])
    dataset = dataset[dataset['answer'].notnull()]
    # shuffle
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    train_samples = int(train_ratio * len(dataset))
    tune_samples = int(tune_ratio * len(dataset))

    prepare_dataset(dataset[0:train_samples], 'train', out_dir)
    prepare_dataset(dataset[train_samples:train_samples + tune_samples], 'tune', out_dir)
    prepare_dataset(dataset[train_samples + tune_samples:], 'test', out_dir)


def prepare_dataset(dataset, type, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, type + '_context.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            br = preprocess(row['br'])
            f.write(br + '\n')

    with open(os.path.join(out_dir, type + '_answer.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            br = preprocess(row['answer'])
            f.write(br + '\n')

    with open(os.path.join(out_dir, type + '_question.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            br = preprocess(row['question'])
            f.write(br + '\n')

    with open(os.path.join(out_dir, type + '_ids.txt'), 'w') as f:
        for index, row in dataset.iterrows():
            br = preprocess_id(row['issue_id'])
            f.write(br + '\n')


def preprocess(text):
    if len(text) < 10:
        print('Suspiciously short text {0}'.format(text))
    return text.replace('\r\n', ' ').replace('\n', ' ')


def preprocess_id(id):
    tokens = id.split('/')
    repo = tokens[-3]
    no = tokens[-1]
    return repo + '_' + no


if __name__ == '__main__':
    run(sys.argv[1:])
