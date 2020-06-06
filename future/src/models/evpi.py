import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from scipy import spatial

import dataset as dataset

import logging

logging.basicConfig(level=logging.INFO)


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


def get_device(cuda):
    if cuda is True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def compute_a_cap(answer, w2v_model):
    words = [w2v_model.index2word[index] for index in answer.numpy().reshape(-1)]
    mean_vec = np.mean(w2v_model[words], axis=0)
    return torch.tensor(mean_vec.reshape(1, len(mean_vec)), dtype=torch.float)


def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def run_evaluation(net, device, w2v_model, test_loader):
    logging.info('Running evaluation...')
    results = {}
    with torch.no_grad():
        for data in test_loader:
            if device.type != 'cpu':
                posts, questions = data['post'].to(device), data['question'].to(device)
            else:
                posts = data['post']
                questions = data['question']

            postid = data['postid'][0]
            answers = data['answer']
            posts_origin = data['post_origin'][0]
            answers_origin = data['answer_origin'][0]
            questions_origin = data['question_origin'][0]

            outputs = net(posts, questions)
            if device.type != 'cpu':
                outputs = outputs.cpu()

            a_cap = compute_a_cap(answers, w2v_model).numpy()

            sim = cosine_similarity(a_cap, outputs)

            score = sim * data['utility']
            if postid not in results:
                results[postid] = (posts_origin, list())
            results[postid][1].append((score, questions_origin, answers_origin))
    return results


def evpi(w2v_model, post_tsv, qa_tsv, n_epoch, cuda):
    device = get_device(cuda)
    print('Running on {0}'.format(device))

    net = EvpiModel(w2v_model.vectors)
    net.to(device)

    train_loader, test_loader = dataset.get_datasets(post_tsv, qa_tsv, w2v_model.vocab)

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(n_epoch):
        logging.info('Epoch {0}/{1}'.format((epoch + 1), n_epoch))
        loss_sum = 0.0
        for data in train_loader:
            # compute a_cap and send it to device so it can be used for back propagationgit pu
            answers = data['answer']
            a_cap = compute_a_cap(answers, w2v_model)

            if device.type != 'cpu':
                posts, questions, a_cap = data['post'].to(device), data['question'].to(device), a_cap.to(device)
            else:
                posts = data['post']
                questions = data['question']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(posts, questions)

            loss = loss_function(outputs, a_cap)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        logging.info('Loss: {0}'.format(loss_sum))

    results = run_evaluation(net, device, w2v_model, test_loader)
    return results
