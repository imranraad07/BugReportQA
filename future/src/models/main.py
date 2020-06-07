import sys
import os

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../pattern_classification'))

import argparse
import evpi
import evpi_batch
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-tsv', help='File path to post_tsv produced by Lucene', required=True)
    parser.add_argument('--qa-tsv', help='File path to qa_tsv produced by Lucene', required=True)
    parser.add_argument('--utility-tsv', help='File path to utility_tsv produced by Lucene', required=True)
    parser.add_argument('--train-ids', help='File path to train ids', required=True)
    parser.add_argument('--test-ids', help='File path to test ids', required=True)
    parser.add_argument('--embeddings', help='File path to embeddings', required=True)
    parser.add_argument('--output-ranking-file', help='Output file to save ranking', required=True)
    parser.add_argument('--max-p-len', help='Max post length. Only when batch_size>1', default=300, type=int)
    parser.add_argument('--max-q-len', help='Max question length. Only when batch_size>1', default=100, type=int)
    parser.add_argument('--max-a-len', help='Max answer length. Only when batch_size>1', default=100, type=int)
    parser.add_argument('--n-epochs', help='Number of epochs', default=10, type=int)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=1)
    parser.add_argument('--device', help='Use \"cuda\" or \"cpu\"', choices=['cuda', 'cpu'])
    return parser.parse_args()


def run():
    args = parse_args()

    logging.info('Running with parameters: {0}'.format(args))

    w2v_model = read_w2v_model(args.embeddings)
    cuda = True if args.device == 'cuda' else False

    if args.batch_size == 1:
        logging.info('Run evpi with batch_size=1')
        results = evpi.evpi(cuda, w2v_model, args)
    else:
        logging.info('Run evpi with batch_size>1')
        results = evpi_batch.evpi(cuda, w2v_model, args)

    save_ranking(args.output_ranking_file, results)


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
    run()
