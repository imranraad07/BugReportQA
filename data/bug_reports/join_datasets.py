import os
import sys
import click
import pandas as pd
import numpy as np

np.random.seed(1234)


@click.command()
@click.option('--input-dir', required=True, default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/bug_reports')
@click.option('--file-prefix', required=True, default='github_data_20')
@click.option('--output-file', required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/datasets/github_partial_2008-2013_part1/dataset.csv')
@click.option('--subset', help='Fraction of original dataset to sample. If subset=1.0, the whole dataset is preserved.')
def join_files(*args, **kwargs):
    dpath = kwargs['input_dir']
    prefix = kwargs['file_prefix']
    out_fpath = kwargs['output_file']

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

    clear_data(out_fpath)

    if kwargs['subset']:
        generate_subset(out_fpath, kwargs['subset'])

    print('Done')


def clear_data(fpath):
    df = pd.read_csv(fpath)
    df = df.drop_duplicates(subset='issue_id')
    df = df[df['answer'].notna()]
    df = df[df['answer'].apply(lambda x: len(x.split()) > 3)]
    df.to_csv(fpath, index=False)


def generate_subset(fpath, subset_ratio):
    df = pd.read_csv(fpath)
    df = df.sample(frac=subset_ratio).reset_index(drop=True)
    df.to_csv(fpath, index=False)


if __name__ == '__main__':
    join_files(sys.argv[1:])
