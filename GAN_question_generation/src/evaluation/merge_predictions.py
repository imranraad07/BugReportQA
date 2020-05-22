import sys
import os
import click


@click.command()
@click.option('--dataset-fpath', required=True)
@click.option('--model-predictions-fpath', required=True)
@click.option('--output-file', required=True)
def run(*args, **kwargs):
    data_dir = kwargs['dataset_fpath']
    ids = read_values(os.path.join(data_dir, 'test_ids.txt'))
    posts = read_values(os.path.join(data_dir, 'test_context.txt'))
    questions = read_values(os.path.join(data_dir, 'test_question.txt'))
    answers = read_values(os.path.join(data_dir, 'test_answer.txt'))

    predictions = read_predictions(kwargs['model_predictions_fpath'])

    with open(kwargs['output_file'], 'w') as f:
        f.write('id,post,question,prediction,answer\n')
        for i in range(0, len(ids)):
            if ids[i] in predictions:
                f.write('{0},{1},{2},{3},{4}\n'.format(ids[i], posts[i], questions[i], predictions[ids[i]], answers[i]))


def read_values(fpath):
    values = list()
    with open(fpath, 'r') as f:
        for line in f.readlines():
            values.append(line.strip())
    return values


def read_predictions(fpath):
    predictions = dict()
    with open(fpath, 'r') as f:
        for line in f.readlines():
            values = line.strip().split(',')
            predictions[values[0]] = ' '.join(values[1:])
    return predictions


if __name__ == '__main__':
    run(sys.argv[1:])
