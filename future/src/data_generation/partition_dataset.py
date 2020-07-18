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
              default='/home/imranm3/projects/BugReportQA/data/datasets/final_dataset/dataset.csv')
@click.option('--out-directory',
              help='Directory to save output dataset files',
              required=True,
              default='/home/imranm3/projects/BugReportQA/data/datasets/final_dataset')
@click.option('--ratios',
              help='Ratio of training/testing dataset. Format: [train_ratio tune_ratio test_ratio]',
              type=(float, float),
              default=[0.8, 0.2],
              required=True)
def run(**kwargs):
    in_dataset = kwargs['in_dataset']
    out_dir = kwargs['out_directory']
    train_ratio, test_ratio = kwargs['ratios']
    if train_ratio +  test_ratio != 1:
        raise ValueError('Ratios of training + tuning + testing datasets should sum up to 1.')

    dataset = pd.read_csv(in_dataset)
    # shuffle
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_samples = int(train_ratio * len(dataset))

    prepare_ids(dataset[0:train_samples], os.path.join(out_dir, 'train_ids.txt'))
    prepare_ids(dataset[train_samples:], os.path.join(out_dir, 'test_ids.txt'))


def prepare_ids(df, fname):
    with open(fname, 'w') as f:
        for index, row in df.iterrows():
            f.write(row['issue_id'] + '\n')


if __name__ == '__main__':
    run(sys.argv[1:])
