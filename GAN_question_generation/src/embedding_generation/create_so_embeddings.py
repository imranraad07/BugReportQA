import os
import cPickle as p
from gensim.models.keyedvectors import KeyedVectors

# download embeddings from: https://github.com/vefstathiou/SO_word2vec
so_embeddings = '/Users/ciborowskaa/Downloads/SO_vectors_200.bin'

repo_path = '/Users/ciborowskaa/VCU/Research/BugReportQA'
project_dir = os.path.join(repo_path, 'GAN_question_generation/embeddings/stack_overflow')

# tokens from Sudha Rao's code
PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOP_token = '<EOP>'
EOS_token = '<EOS>'

def run():
    word_vect = KeyedVectors.load_word2vec_format(so_embeddings, binary=True)
    print('Embeddings loaded')

    vector_list = list()
    #append 0s for special tokens
    for i in range(0, 4):
        vector_list.append([0]*(word_vect.vectors.shape[1]))

    for vector in word_vect.vectors:
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
    for idx, word in enumerate(word_vect.index2word):
        vocab[word] = idx+4

    f = open(os.path.join(project_dir, 'vocab.p'), 'wb')
    p.dump(vocab, f)
    f.close()


if __name__ == '__main__':
    run()
