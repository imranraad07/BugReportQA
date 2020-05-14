import pandas as pd
import click
import sys


@click.command()
@click.option('--data-csv', required=True)
@click.option('--output-fpath', required=True)
@click.option('--triplets', is_flag=True)
def run(*args, **kwargs):
    df = pd.read_csv(kwargs['data_csv'])
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    out_fpath = kwargs['output_fpath']
    if kwargs['triplets'] is not None:
        save_together(df, out_fpath)
    else:
        save_separately(df, out_fpath)


def save_together(df, out_fpath):
    with open(out_fpath, 'w') as f:
        for idx, row in df.iterrows():
            if not isinstance(row['answer'], str):
                continue
            f.write(preprocess(row['post'] + ' '))
            f.write(preprocess(row['question'] + ' '))
            f.write(preprocess(row['answer'] + ' '))
            f.write('\n')


def save_separately(df, out_fpath):
    with open(out_fpath, 'w') as f:
        for idx, row in df.iterrows():
            if not isinstance(row['answer'], str):
                continue
            f.write(preprocess(row['post'] + '\n'))
        for idx, row in df.iterrows():
            if not isinstance(row['answer'], str):
                continue
            f.write(preprocess(row['question'] + '\n'))
        for idx, row in df.iterrows():
            if not isinstance(row['answer'], str):
                continue
            f.write(preprocess(row['answer'] + '\n'))


def preprocess(text):
    text = str(text).lower().strip().replace('\n', ' ')

    # punctuation separation
    text_filtered = ''
    for c in text:
        if c in '!@$%^&*()[]{};:,.\'\"/<>?\|`~-=+':
            text_filtered += ' ' + c +' '
        else:
            text_filtered += c

    # tokens = text_filtered.split(' ')
    # filter stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if w not in stop_words]

    return text_filtered


if __name__ == '__main__':
    run(sys.argv[1:])
