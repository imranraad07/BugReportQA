import os
import cPickle as p
from gensim.models.keyedvectors import KeyedVectors

# download embeddings from: https://github.com/vefstathiou/SO_word2vec
so_embeddings = '/Users/ciborowskaa/Downloads/SO_vectors_200.bin'

repo_path = '/Users/ciborowskaa/VCU/Research/BugReportQA'
project_dir = os.path.join(repo_path, 'GAN_question_generation/embeddings/stack_overflow')


def run():
    word_vect = KeyedVectors.load_word2vec_format(so_embeddings, binary=True)
    print('Embeddings loaded')

    vector_list = list()
    for vector in word_vect.vectors:
        vector_list.append(vector.tolist())

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    f = open(os.path.join(project_dir, 'word_embeddings.p'), 'wb')
    p.dump(vector_list, f)
    f.close()

    vocab = dict()
    for idx, word in enumerate(word_vect.index2word):
        vocab[word] = idx

    f = open(os.path.join(project_dir, 'vocab.p'), 'wb')
    p.dump(vocab, f)
    f.close()


if __name__ == '__main__':
    run()
