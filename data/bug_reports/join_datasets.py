import os
import sys
import click
import pandas as pd


@click.command()
@click.option('--input-dir', required=True, default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/bug_reports')
@click.option('--file-prefix', required=True, default='github_data_20')
@click.option('--output-file', required=True,
              default='/Users/ciborowskaa/VCU/Research/BugReportQA/data/datasets/github_partial_2008-2011/dataset.csv')
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
    with open(out_fpath, 'w') as f:
        f.write(header)
        for line in lines:
            f.write(line + '\n')

    filter_duplicates(out_fpath)
    print('Done')


def filter_duplicates(fpath):
    df = pd.read_csv(fpath)
    df = df.drop_duplicates(subset='issue_id')
    df.to_csv(fpath, index=False)


if __name__ == '__main__':
    join_files(sys.argv[1:])
