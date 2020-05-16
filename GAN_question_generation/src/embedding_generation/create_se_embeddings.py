import os
import cPickle as p
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

# if you dont have nltk.stopwords uncomment 2 lines below
# import nltk
# nltk.download('stopwords')


se_embeddings = '/Users/ciborowskaa/Downloads/stackexchange_embeddings/embeddings/word_embeddings.p'
se_vocab = '/Users/ciborowskaa/Downloads/stackexchange_embeddings/embeddings/vocab.p'

repo_path = '/Users/ciborowskaa/VCU/Research/BugReportQA'
project_dir = os.path.join(repo_path, 'GAN_question_generation/embeddings/stackexchange')
dataset_path = os.path.join(repo_path, 'data/bug_reports/github_dataset_partial.csv')

# tokens from Sudha Rao's code
PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOP_token = '<EOP>'
EOS_token = '<EOS>'
UNK_token = '<unk>'
SPECIFIC_token = '<specific>'
GENERIC_token = '<generic>'


def run():
    word_embeddings = p.load(open(se_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(se_vocab, 'rb'))
    index2word = reverse_dict(word2index)

    print('SE Embeddings loaded')

    br_vocab = get_br_vocab(dataset_path)

    vector_list = list()
    # append 0s for special tokens
    for i in range(0, 7):
        vector_list.append([0] * word_embeddings.shape[1])

    for idx in range(0, len(index2word)):
        word = index2word[idx]
        if word in br_vocab:
            vector_list.append(word_embeddings[idx].tolist())

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    f = open(os.path.join(project_dir, 'word_embeddings.p'), 'wb')
    p.dump(vector_list, f)
    f.close()

    vocab = dict()
    vocab[PAD_token] = 0
    vocab[SOS_token] = 1
    vocab[EOP_token] = 2
    vocab[EOS_token] = 3
    vocab[UNK_token] = 4
    vocab[SPECIFIC_token] = 5
    vocab[GENERIC_token] = 6
    idx = 7
    for index in index2word:
        word = index2word[index]
        if word in br_vocab:
            vocab[word] = idx
            idx += 1

    f = open(os.path.join(project_dir, 'vocab.p'), 'wb')
    p.dump(vocab, f)
    f.close()


def get_br_vocab(dataset_path):
    vocab = set()
    data = pd.read_csv(dataset_path)
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    for index, row in data.iterrows():
        vocab = update_vocab(row['post'], vocab)
        vocab = update_vocab(row['question'], vocab)
        vocab = update_vocab(row['answer'], vocab)
    return vocab


def update_vocab(line, vocab):
    line = str(line).lower().strip().replace('\n', ' ')

    # punctuation removal
    line_filtered = ''
    for c in line:
        if c in '!@$%^&*()[]{};:,./<>?\|`~-=':
            line_filtered += ' ' + c + ' '
        else:
            line_filtered += c

    tokens = line_filtered.split(' ')
    # filter stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if w not in stop_words]
    vocab.update(tokens)
    return vocab


def reverse_dict(word2index):
    index2word = {}
    for w, ix in word2index.iteritems():
        index2word[ix] = w
    return index2word


if __name__ == '__main__':
    run()
