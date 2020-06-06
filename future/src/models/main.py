import sys
import os

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../pattern_classification'))

import click
import evpi
import evpi_batch
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


@click.command()
@click.option('--post-tsv', help='File path to post_tsv produced by Lucene', required=True)
@click.option('--qa-tsv', help='File path to qa_tsv produced by Lucene', required=True)
@click.option('--train-ids', help='File path to train ids', required=True)
@click.option('--test-ids', help='File path to test ids', required=True)
@click.option('--embeddings', help='File path to embeddings', required=True)
@click.option('--output-ranking-file', help='Output file to save ranking', required=True)
@click.option('--max-p-len', help='Max post length. Only when batch_size>1', default=300)
@click.option('--max-q-len', help='Max question length. Only when batch_size>1', default=100)
@click.option('--max-a-len', help='Max answer length. Only when batch_size>1', default=100)
@click.option('--n-epochs', help='Number of epochs', default=10)
@click.option('--batch-size', help='Batch size', type=int, default=1)
@click.option('--device', help='Use \"cuda\" or \"cpu\"', type=click.Choice(['cuda', 'cpu']))
def run(**kwargs):
    logging.info('Running with parameters: {0}'.format(kwargs))

    w2v_model = read_w2v_model(kwargs['embeddings'])
    post_tsv = kwargs['post_tsv']
    qa_tsv = kwargs['qa_tsv']
    n_epoch = kwargs['n_epochs']
    batch_size = kwargs['batch_size']
    train_ids = kwargs['train_ids']
    test_ids = kwargs['test_ids']
    cuda = True if kwargs['device'] == 'cuda' else False

    if kwargs['batch_size'] == 1:
        logging.info('Run evpi with batch_size=1')
        results = evpi.evpi(w2v_model, post_tsv, qa_tsv, train_ids, test_ids, n_epoch, cuda)
    else:
        logging.info('Run evpi with batch_size>1')
        max_p_len = kwargs['max_p_len']
        max_q_len = kwargs['max_q_len']
        max_a_len = kwargs['max_a_len']
        results = evpi_batch.evpi(w2v_model, post_tsv, qa_tsv, train_ids, test_ids, n_epoch, batch_size, cuda,
                                  max_p_len, max_q_len, max_a_len)

    save_ranking(kwargs['output_ranking_file'], results)


def read_w2v_model(path_in):
    path_out = '/'.join(path_in.split('/')[:-1]) + '/w2v_vectors.txt'
    glove2word2vec(path_in, path_out)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_out)
    if '<PAD>' not in w2v_model or w2v_model.vocab['<PAD>'].index != 0:
        raise ValueError('No <PAD> token in embeddings! Provide embeddings with <PAD> token.')
    return w2v_model


def save_ranking(output_file, results):
    with open(output_file, 'w') as f:
        f.write('postid,post,correct_q,correct_a,' + ','.join(['q{0},a{0}'.format(i) for i in range(1, 11)]) + '\n')
        for postid in results:
            post, values, correct = results[postid]
            f.write('{0},{1},{2},{3},'.format(postid, post.replace(',', ' '), correct[0].replace(',', ' '),
                                              correct[1].replace(',', ' ')))

            values = sorted(values, key=lambda x: x[0], reverse=True)
            for score, question, answer in values:
                f.write('{0},{1},'.format(question.replace(',', ' '), answer.replace(',', ' ')))
            f.write('\n')


if __name__ == '__main__':
    run(sys.argv[1:])
