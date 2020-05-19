import os
import sys
import click
from nltk.translate.bleu_score import corpus_bleu


@click.command()
@click.option('--ref-file', help='Path to file with correct questions', required=True)
@click.option('--q-file', help='Path to file with questions', required=True)
def run(*args, **kwargs):
    refs = extract_ref(kwargs['ref_file'])
    data = extract_data(kwargs['q_file'])
    score = corpus_bleu(refs, data)

    print('BLEU score {0}'.format(score))


def extract_ref(fpath):
    ref_corpus = list()
    with open(fpath, 'r') as f:
        for line in f.readlines():
            refs = list()
            for ref in line.split('|'):
                tokens = tokenize(ref)
                refs.append(tokens)
            ref_corpus.append(refs)
    return ref_corpus


def extract_data(fpath):
    corpus = list()
    with open(fpath, 'r') as f:
        for line in f.readlines():
            tokens = tokenize(line)
            corpus.append(tokens)
    return corpus


def tokenize(text):
    line = str(text).lower().strip()
    line = line.replace('<unk>', ' ').replace('<EOS>', ' ')
    # punctuation tokenizing
    line_filtered = ''
    for c in line:
        if c in '!@$%^&*()[]{};:,.<>/?\|`~-=':
            line_filtered += ' '
        else:
            line_filtered += c

    return line_filtered.split(' ')


if __name__ == '__main__':
    run(sys.argv[1:])
