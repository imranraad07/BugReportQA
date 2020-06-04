import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import models.dataset as dataset
import models.calculator as calc

import logging

logging.basicConfig(level=logging.INFO)
CUDA = False

"""
TODO Big things:
1. Evaluation on test set
2. Ranking
3. (Done: make sure ids in the dataset are properly divided! One tsv row = 10 data points (p, q_i, a_i))
4. Padding for batch processing - do it in a new file; it requires more modifications

TODO Small things:
1. Make sure diff computation makes sense.
2. Move utility calculator as a part of dataset building set - OB is not changing.

Nice to have:
1. Evaluation on validation set and saving model that performs best on validation set.
   We can use that model later on for evaluation with test set.
"""


class EvpiModel(nn.Module):

    def __init__(self, weights, hidden_dim=256):
        super(EvpiModel, self).__init__()

        # layer 1 - embeddings
        weights = torch.FloatTensor(weights)
        self.emb_layer = nn.Embedding.from_pretrained(weights)
        self.emb_layer.requires_grad = False

        # layer 2 - LSTMs
        self.p_lstm = nn.LSTM(input_size=self.emb_layer.embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.q_lstm = nn.LSTM(input_size=self.emb_layer.embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # layer 3 - dense layer
        self.layer1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.layer2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Linear(2 * hidden_dim, self.emb_layer.embedding_dim)
        self.layer4 = nn.Linear(self.emb_layer.embedding_dim, self.emb_layer.embedding_dim)

    def forward(self, post, question):
        p_emb_out = self.emb_layer(post)
        q_emb_out = self.emb_layer(question)
        p_lstm_out, test = self.p_lstm(p_emb_out)
        q_lstm_out, test = self.q_lstm(q_emb_out)
        lstm_avg = torch.cat((p_lstm_out.mean(dim=1), q_lstm_out.mean(dim=1)), 1)
        x = F.relu(self.layer1(lstm_avg))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        answer_representation = self.layer4(x)
        return answer_representation


def get_device():
    cuda = CUDA and torch.cuda.is_available()
    if cuda is True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def compute_a_cap(answer, w2v_model):
    words = [w2v_model.index2word[index] for index in answer.numpy().reshape(-1)]
    mean_vec = np.mean(w2v_model[words], axis=0)
    return torch.tensor(mean_vec.reshape(1, len(mean_vec)), dtype=torch.float)


def evpi(vector_fpath, post_tsv, qa_tsv, n_epochs):
    glove2word2vec(vector_fpath, '../../embeddings_damevski/w2v_vectors.txt')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../../embeddings_damevski/w2v_vectors.txt')
    net = EvpiModel(w2v_model.vectors)
    calculator = calc.Calculator()

    device = get_device()
    net.to(device)
    print('Running on {0}'.format(device))

    train_loader, test_loader = dataset.get_datasets(post_tsv, qa_tsv, w2v_model.vocab, batch_size=1)

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [post, question, answer, label]
            if CUDA:
                posts, questions = data['post'].to(device), data['question'].to(device)
            else:
                posts = data['post']
                questions = data['question']

            answers = data['answer']
            posts_origin = data['post_origin']
            answers_origin = data['answer_origin']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(posts, questions)
            a_cap = compute_a_cap(answers, w2v_model)

            loss = loss_function(outputs, a_cap)
            loss += (1 - calculator.utility(answers_origin, posts_origin))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

    # evaluate with test set
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # get the inputs; data is a list of [post, question, answer, label]
            if CUDA:
                posts, questions = data['post'].to(device), data['question'].to(device)
                answers = data['answer']
            else:
                posts = data['post']
                questions = data['question']
                answers = data['answer']

            outputs = net(posts, questions)
            a_cap = compute_a_cap(answers, w2v_model)
            loss = loss_function(outputs, a_cap)


if __name__ == '__main__':
    evpi('../../embeddings_damevski/vectors.txt',
         '../../data/github_partial_2008-2013_part1_small/post_data.tsv',
         '../../data/github_partial_2008-2013_part1_small/qa_data.tsv',
         100)
