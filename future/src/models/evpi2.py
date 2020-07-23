import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from scipy import spatial

import dataset as dataset

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

        # layer 3 - dense layer
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.emb_layer.embedding_dim)
        self.layer4 = nn.Linear(self.emb_layer.embedding_dim, self.emb_layer.embedding_dim)

    def forward(self, post):
        p_emb_out = self.emb_layer(post)
        p_lstm_out, test = self.p_lstm(p_emb_out)
        x = F.relu(self.layer1(p_lstm_out.mean(dim=1)))
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


def compute_qa_cap(question, answer, w2v_model):
    a_words = [w2v_model.index2word[index] for index in answer.numpy().reshape(-1)]
    q_words = [w2v_model.index2word[index] for index in question.numpy().reshape(-1)]
    mean_vec = np.mean(np.concatenate((w2v_model[a_words],w2v_model[q_words])), axis=0)
    return torch.tensor(mean_vec.reshape(1, len(mean_vec)), dtype=torch.float)


def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def run_evaluation(net, device, w2v_model, test_loader):
    logging.info('Running evaluation...')
    results = {}
    with torch.no_grad():
        for data in test_loader:
            if device.type != 'cpu':
                posts = data['post'].to(device)
            else:
                posts = data['post']

            postid = data['postid'][0]
            questions = data['question']
            answers = data['answer']
            posts_origin = data['post_origin'][0]
            answers_origin = data['answer_origin'][0]
            questions_origin = data['question_origin'][0]
            label = data['label'][0]

            outputs = net(posts)
            if device.type != 'cpu':
                outputs = outputs.cpu()

            a_cap = compute_qa_cap(questions, answers, w2v_model).numpy()
            sim = cosine_similarity(a_cap, outputs)

            score = sim * data['utility']
            if postid not in results:
                results[postid] = (posts_origin, list(), list())
            results[postid][1].append((score, questions_origin, answers_origin))
            if label == 1:
                results[postid][2].extend([questions_origin, answers_origin])

    return results


def evpi(cuda, w2v_model, args):
    device = get_device(cuda)
    logging.info('Running on {0}'.format(device))

    net = EvpiModel(w2v_model.vectors)
    net.to(device)

    train_loader, test_loader = dataset.get_datasets(w2v_model.vocab, args)

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(args.n_epochs):
        logging.info('Epoch {0}/{1}'.format((epoch + 1), args.n_epochs))
        loss_sum = 0.0
        for data in train_loader:
            # compute a_cap and send it to device so it can be used for back propagation
            answers = data['answer']
            questions = data['question']
            a_cap = compute_qa_cap(questions, answers, w2v_model)

            if device.type != 'cpu':
                posts,  a_cap = data['post'].to(device), a_cap.to(device)
            else:
                posts = data['post']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(posts)

            loss = loss_function(outputs, a_cap)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        logging.info('Loss: {0}'.format(loss_sum / len(train_loader)))

    results = run_evaluation(net, device, w2v_model, test_loader)
    return results
