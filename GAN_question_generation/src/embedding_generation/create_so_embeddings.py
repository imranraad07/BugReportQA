import os
import cPickle as p
import pandas as pd
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors

# if you dont have nltk.stopwords uncomment 2 lines below
# import nltk
# nltk.download('stopwords')

# download embeddings from: https://github.com/vefstathiou/SO_word2vec
so_embeddings = '/Users/ciborowskaa/Downloads/SO_vectors_200.bin'

repo_path = '/Users/ciborowskaa/VCU/Research/BugReportQA'
project_dir = os.path.join(repo_path, 'GAN_question_generation/embeddings/stack_overflow')
dataset_path = os.path.join(repo_path, 'data/bug_reports/github_data.csv')

# tokens from Sudha Rao's code
PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOP_token = '<EOP>'
EOS_token = '<EOS>'
UNK_token = '<unk>'
SPECIFIC_token = '<specific>'
GENERIC_token = '<generic>'


def run():
    word_vect = KeyedVectors.load_word2vec_format(so_embeddings, binary=True)
    print('SO Embeddings loaded')

    br_vocab = get_br_vocab(dataset_path)

    vector_list = list()
    # append 0s for special tokens
    for i in range(0, 7):
        vector_list.append([0] * (word_vect.vectors.shape[1]))

    for idx, vector in enumerate(word_vect.vectors):
        word = word_vect.index2word[idx]
        if word in br_vocab:
            vector_list.append(vector.tolist())

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
    for word in word_vect.index2word:
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
    line = str(line).lower().strip()

    # punctuation removal
    line_filtered = ''
    for c in line:
        if c in '!@$%^&*()[]{};:,./<>?\|`~-=':
            line_filtered += ' '
        else:
            line_filtered += c

    tokens = line_filtered.split(' ')
    # filter stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    vocab.update(tokens)
    return vocab


if __name__ == '__main__':
    run()
