import click
import pandas as pd
import os
import sys
import numpy

# for shuffling
numpy.random.seed(1234)


@click.command()
@click.option('--in-dataset',
              help='Input dataset file',
              required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/datasets/github_partial_2008-2011/dataset.csv')
@click.option('--out-directory',
              help='Directory to save output dataset files',
              required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/datasets/github_partial_2008-2011')
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

    dataset = pd.read_csv(in_dataset)
    dataset = dataset[dataset['answer'].notnull()]
    # shuffle
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_samples = int(train_ratio * len(dataset))
    tune_samples = int(tune_ratio * len(dataset))

    prepare_ids(dataset[0:train_samples], os.path.join(out_dir, 'train_ids.txt'))
    prepare_ids(dataset[train_samples:train_samples + tune_samples], os.path.join(out_dir, 'tune_ids.txt'))
    prepare_ids(dataset[train_samples + tune_samples:], os.path.join(out_dir, 'test_ids.txt'))


def prepare_ids(df, fname):
    with open(fname, 'w') as f:
        for index, row in df.iterrows():
            f.write(row['issue_id'] + '\n')


if __name__ == '__main__':
    run(sys.argv[1:])
