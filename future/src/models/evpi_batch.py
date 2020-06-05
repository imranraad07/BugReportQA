import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy import spatial

import dataset_batch as dataset

import logging

logging.basicConfig(level=logging.INFO)
CUDA = False


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

    def forward(self, post, post_lengths, question, question_lengths):
        # sort data
        post, post_lengths, post_sorted_id = self.sort_batch(post, post_lengths)
        question, question_lengths, question_sorted_id = self.sort_batch(question, question_lengths)

        # process posts
        p_emb_out = self.emb_layer(post)
        p_emb_out = pack_padded_sequence(p_emb_out, post_lengths, batch_first=True)
        p_lstm_out, _ = self.p_lstm(p_emb_out)
        p_lstm_out, _ = pad_packed_sequence(p_lstm_out, batch_first=True)
        p_mean = p_lstm_out.mean(dim=1)

        # process questions
        q_emb_out = self.emb_layer(question)
        q_emb_out = pack_padded_sequence(q_emb_out, question_lengths, batch_first=True)
        q_lstm_out, _ = self.q_lstm(q_emb_out)
        q_lstm_out, _ = pad_packed_sequence(q_lstm_out, batch_first=True)
        q_mean = q_lstm_out.mean(dim=1)

        # restore ordering
        p_mean, post_lengths = self.restore_ordering(p_mean, post_lengths, post_sorted_id)
        q_mean, question_lengths = self.restore_ordering(q_mean, question_lengths, question_sorted_id)

        # linear layers
        lstm_avg = torch.cat((p_mean, q_mean), 1)
        x = F.relu(self.layer1(lstm_avg))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        answer_representation = self.layer4(x)
        return answer_representation

    def sort_batch(self, data, seq_len):
        """ Sort the data (B, T, D) and sequence lengths """
        sorted_seq_len, sorted_idx = seq_len.sort(0, descending=True)
        sorted_data = data[sorted_idx]
        return sorted_data, sorted_seq_len, sorted_idx

    def restore_ordering(self, data, seq_len, sorted_idx):
        """ Restore original ordering of data based on sorted ids """
        sorted_sorted_id, initial_id = sorted_idx.sort(0, descending=False)
        sorted_data = data[initial_id]
        sorted_len = seq_len[initial_id]
        return sorted_data, sorted_len


def get_device():
    if CUDA is True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def compute_a_cap(answer, w2v_model):
    answer = answer.numpy()
    a_cap = np.ndarray((answer.shape[0], w2v_model.vectors.shape[1]))
    for idx, tensor in enumerate(answer):
        words = [w2v_model.index2word[index] for index in tensor]
        a_cap[idx] = np.mean(w2v_model[words], axis=0)
    return torch.tensor(a_cap, dtype=torch.float)


def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def run_evaluation(net, device, w2v_model, test_loader):
    results = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if CUDA and torch.cuda.is_available():
                posts, post_len, questions, q_len, a_cap = data['post'].to(device), data['post_len'].to(device), \
                                                           data['question'].to(device), data['q_len'].to(device), \
                                                           a_cap.to(device)
            else:
                posts = data['post']
                questions = data['question']
                post_len = data['post_len']
                q_len = data['q_len']

            postid = data['postid']
            answers = data['answer']
            utility = data['utility'].numpy()
            posts_origin = data['post_origin']
            answers_origin = data['answer_origin']
            questions_origin = data['question_origin']

            outputs = net(posts, post_len, questions, q_len)
            if device.type != 'cpu':
                outputs = outputs.cpu()

            outputs = outputs.numpy()

            a_cap = compute_a_cap(answers, w2v_model).numpy()

            for i in range(0, test_loader.batch_size):
                sim = cosine_similarity(a_cap[i], outputs[i])

                score = sim * utility[i]
                if postid[i] not in results:
                    results[postid[i]] = (posts_origin[i], list())
                results[postid[i]][1].append((score, questions_origin[i], answers_origin[i]))
    return results


def create_ranking(output_file, results):
    with open(output_file, 'w') as f:
        f.write('postid,post,' + ','.join(['q{0},a{0}'.format(i) for i in range(1, 11)]) + '\n')
        for postid in results:
            post, values = results[postid]
            f.write('{0},{1},'.format(postid, post.replace(',', ' ')))

            values = sorted(values, key=lambda x: x[0], reverse=True)
            for score, question, answer in values:
                f.write('{0},{1},'.format(question.replace(',', ' '), answer.replace(',', ' ')))
            f.write('\n')


def evpi(vector_fpath, post_tsv, qa_tsv, ranking_output, n_epochs):
    glove2word2vec(vector_fpath, '../../embeddings_damevski/w2v_vectors.txt')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../../embeddings_damevski/w2v_vectors.txt')
    if '<PAD>' not in w2v_model or w2v_model.vocab['<PAD>'].index != 0:
        raise ValueError('No <PAD> token in embeddings! Provide embeddings with <PAD> token.')

    net = EvpiModel(w2v_model.vectors)

    device = get_device()
    net.to(device)
    print('Running on {0}'.format(device))

    train_loader, test_loader = dataset.get_datasets(post_tsv, qa_tsv, w2v_model.vocab, batch_size=2)

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_loader):
            # compute a_cap and send it to device so it can be used for back propagation
            answers = data['answer']
            a_cap = compute_a_cap(answers, w2v_model)

            if CUDA and torch.cuda.is_available():
                posts, post_len, questions, q_len, a_cap = data['post'].to(device), data['post_len'].to(device), \
                                                           data['question'].to(device), data['q_len'].to(device), \
                                                           a_cap.to(device)
            else:
                posts = data['post']
                questions = data['question']
                post_len = data['post_len']
                q_len = data['q_len']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(posts, post_len, questions, q_len)

            loss = loss_function(outputs, a_cap)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

    results = run_evaluation(net, device, w2v_model, test_loader)
    create_ranking(ranking_output, results)


if __name__ == '__main__':
    evpi('../../embeddings_damevski/vectors_pad.txt',
         '../../data/github_partial_2008-2013_part1_small/post_data.tsv',
         '../../data/github_partial_2008-2013_part1_small/qa_data.tsv',
         'ranking.csv',
         1)
