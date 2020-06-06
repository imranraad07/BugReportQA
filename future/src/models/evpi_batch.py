import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from scipy import spatial

import dataset_batch as dataset

import logging


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


def get_device(cuda):
    if cuda is True:
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
    logging.info('Running evaluation...')
    results = {}
    with torch.no_grad():
        for data in test_loader:
            if device.type != 'cpu':
                posts, post_len, questions, q_len, a_cap = data['post'].to(device), data['post_len'].to(device), \
                                                           data['question'].to(device), data['q_len'].to(device), \
                                                           a_cap.to(device)
            else:
                posts = data['post']
                questions = data['question']
                post_len = data['post_len']
                q_len = data['q_len']

            postids = data['postid']
            answers = data['answer']
            utility = data['utility'].numpy()
            posts_origin = data['post_origin']
            answers_origin = data['answer_origin']
            questions_origin = data['question_origin']
            labels = data['label']

            outputs = net(posts, post_len, questions, q_len)
            if device.type != 'cpu':
                outputs = outputs.cpu()

            outputs = outputs.numpy()

            a_cap = compute_a_cap(answers, w2v_model).numpy()

            for idx in range(0, test_loader.batch_size):
                postid = postids[idx]
                sim = cosine_similarity(a_cap[idx], outputs[idx])

                score = sim * utility[idx]
                if postid not in results:
                    results[postid] = (posts_origin[idx], list(), list())
                results[postid][1].append((score, questions_origin[idx], answers_origin[idx]))
                if labels[idx] == 1:
                    results[postid][2].extend([questions_origin[idx], answers_origin[idx]])

    return results


def evpi(w2v_model, post_tsv, qa_tsv, n_epoch, batch_size, cuda, max_p_len, max_q_len, max_a_len):
    device = get_device(cuda)
    logging.info('Running on {0}'.format(device))

    net = EvpiModel(w2v_model.vectors)
    net.to(device)

    train_loader, test_loader = dataset.get_datasets(post_tsv, qa_tsv, w2v_model.vocab, batch_size=batch_size,
                                                     max_post_len=max_p_len, max_q_len=max_q_len, max_a_len=max_a_len)

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(n_epoch):
        logging.info('Epoch {0}/{1}'.format((epoch + 1), n_epoch))
        loss_sum = 0.0
        for data in train_loader:
            # compute a_cap and send it to device so it can be used for back propagation
            answers = data['answer']
            a_cap = compute_a_cap(answers, w2v_model)

            if device.type != 'cpu':
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

        logging.info('Loss: {0}'.format(loss_sum))

    results = run_evaluation(net, device, w2v_model, test_loader)
    return results


if __name__ == '__main__':
    evpi('../../embeddings_damevski/vectors_pad.txt',
         '../../data/github_partial_2008-2013_part1_small/post_data.tsv',
         '../../data/github_partial_2008-2013_part1_small/qa_data.tsv',
         1)
