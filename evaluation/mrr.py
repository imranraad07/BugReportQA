import sys
import click
import pandas as pd


@click.command()
@click.option('--results-file', required=True)
@click.option('--output-file', required=True)
def run(**kwargs):
    df = pd.read_csv(kwargs['results_file'], sep=',', index_col=False)

    ranks = list()
    for index, row in df.iterrows():
        correct = row['correct_question']
        for i in range(1, 11):
            q = row['q' + str(i)]
            if q == correct:
                ranks.append(1.0 / float(i))
                break
            if i == 10:
                raise ValueError('This shouldnt happen. Row is\n{0}'.format(row))

    mrr = sum(ranks) / float(len(ranks))
    print('MRR {0}'.format(mrr))


if __name__ == '__main__':
    run(sys.argv[1:])
