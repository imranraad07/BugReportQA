import click
import sys
import pandas as pd


@click.command()
@click.option('--results-file', required=True)
@click.option('--output-refs', required=True)
@click.option('--output-questions', required=True)
def run(**kwargs):
    results = kwargs['results_file']
    out_refs = kwargs['output_refs']
    out_questions = kwargs['output_questions']

    data4ranking(results, out_refs, out_questions)


def data4ranking(in_refs, out_refs, out_questions):
    new_refs = open(out_refs, 'w')
    new_q = open(out_questions, 'w')
    df = pd.read_csv(in_refs, sep=',', index_col=False)
    for index, row in df.iterrows():
        new_refs.write(row['q1'] + '\n')
        new_q.write(row['correct_question'] + '\n')

    new_refs.close()
    new_q.close()

    print('Done')


if __name__ == '__main__':
    run(sys.argv[1:])
