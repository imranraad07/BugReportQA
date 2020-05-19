import click
import sys


@click.command()
@click.command('--input-refs', required=True,
               help='File with correct questions. When method=ranking, the file contains also ranked questions.')
@click.command('--input-questions', help='File with generated/ranked questions. Use only when method=GAN.')
@click.command('--output-refs', required=True)
@click.command('--output-questions', required=True)
@click.command('--method', required=True, type=click.Choice(['GAN', 'ranking']))
def run(*args, **kwargs):
    method = kwargs['method']
    in_refs = kwargs['input_refs']
    out_refs = kwargs['output_refs']
    out_questions = kwargs['output_questions']

    if method == 'GAN':
        print('GAN output is compatible by default. Simply copy input to output')
        in_questions = kwargs['input_questions']
        data4GAN(in_refs, out_refs, in_questions, out_questions)
    elif method == 'ranking':
        data4ranking(in_refs, out_refs, out_questions)
    else:
        raise ValueError('Unknown method {0}'.format(method))


def data4GAN(in_refs, out_refs, in_questions, out_questions):
    new_refs = open(out_refs, 'w')
    with open(in_refs, 'r') as f:
        new_refs.write(f.read())
    new_refs.close()

    new_q = open(out_questions, 'w')
    with open(in_questions, 'r') as f:
        new_q.write(f.read())
    new_q.close()

    print('Done')


def data4ranking(in_refs, out_refs, out_questions):
    new_refs = open(out_refs, 'w')
    new_q = open(out_questions, 'w')
    with open(in_refs, 'r') as f:
        # skip header
        # id, <10 predictions ordered by the numbers in the output file>, correct question
        f.readline()
        for line in f.readlines():
            values = line.strip().split(',')
            new_refs.write(values[11] + '\n')
            new_q.write(values[1] + '\n')

    new_refs.close()
    new_q.close()

    print('Done')


if __name__ == '__main__':
    run(sys.argv[1:])
