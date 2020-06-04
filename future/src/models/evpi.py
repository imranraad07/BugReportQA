import pandas as pd
import numpy as np

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import spacy
from spacy.matcher import Matcher
from difflib import Differ

import preprocessing as pp
import pattern_classification.observed_behavior_rule as ob

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

class GithubDataset(Dataset):

    def __init__(self, post_tsv, qa_tsv, train, transform=None):
        self.transform = transform
        self.dataset, self.index2postid = self._build_dataset(post_tsv, qa_tsv, train)

    def _build_dataset(self, post_tsv, qa_tsv, train):
        posts = pd.read_csv(post_tsv, sep='\t')[:13]
        qa = pd.read_csv(qa_tsv, sep='\t')[:13]
        data = {'post_origin': list(),
                'question_origin': list(),
                'answer_origin': list(),
                'label': list(),
                'id': list()}

        index2postid = dict()
        for idx, row in posts.iterrows():
            # TODO: this should be covered when building dataset
            if str(row['post']) == 'nan':
                continue
            id_str = row['postid']
            if id_str not in index2postid:
                index2postid[id_str] = len(index2postid)
            id = index2postid[id_str]
            for i in range(1, 11):
                data['post_origin'].append(row['title'] + ' ' + row['post'])
                data['id'].append(id)
                data['answer_origin'].append(qa.iloc[idx]['a' + str(i)])
                data['question_origin'].append(qa.iloc[idx]['q' + str(i)])
                if i == 1:
                    data['label'].append(1)
                else:
                    data['label'].append(0)
        dataset = pd.DataFrame(data)

        instances_no = len(dataset) / 10
        train_instances = int(instances_no * 0.9)
        if train is True:
            dataset = dataset[:(train_instances * 10)]
        else:
            dataset = dataset[(train_instances * 10):].reset_index(drop=True)
        return dataset, index2postid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset.iloc[idx]
        sample = self._to_dict(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _to_dict(self, sample):
        new_sample = {'post_origin': sample['post_origin'],
                      'id': sample['id'],
                      'question_origin': sample['question_origin'],
                      'answer_origin': sample['answer_origin'],
                      'label': sample['label']}
        return new_sample


class Preprocessing(object):

    def __call__(self, sample):
        sample['post'] = pp.clear_text(sample['post_origin'], keep_punctuation=True)
        sample['question'] = pp.clear_text(sample['question_origin'], keep_punctuation=True)
        sample['answer'] = pp.clear_text(sample['answer_origin'], keep_punctuation=True)
        return sample


class Word2Idx(object):

    def __init__(self, word2index):
        self.word2index = word2index

    def __call__(self, sample):
        sample['post'] = self._process(sample['post'])
        sample['answer'] = self._process(sample['answer'])
        sample['question'] = self._process(sample['question'])
        return sample

    def _process(self, text):
        idxs = [self.word2index[w].index for w in text.split() if w in self.word2index]
        return idxs


class ToTensor(object):

    def __call__(self, sample):
        sample['post'] = torch.tensor(sample['post'], dtype=torch.long)
        sample['question'] = torch.tensor(sample['question'], dtype=torch.long)
        sample['answer'] = torch.tensor(sample['answer'], dtype=torch.long)
        sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
        sample['id'] = torch.tensor(sample['id'], dtype=torch.int32)
        return sample


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


def get_datasets(post_tsv, qa_tsv, word2index, batch_size=256, shuffle=True):
    preprocess = Preprocessing()
    w2idx = Word2Idx(word2index)
    to_tensor = ToTensor()
    train_dataset = GithubDataset(post_tsv, qa_tsv, transform=transforms.Compose([preprocess, w2idx, to_tensor]),
                                  train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    test_dataset = GithubDataset(post_tsv, qa_tsv, transform=transforms.Compose([preprocess, w2idx, to_tensor]),
                                 train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return train_loader, test_loader


def compute_a_cap(answer, w2v_model):
    words = [w2v_model.index2word[index] for index in answer.numpy().reshape(-1)]
    mean_vec = np.mean(w2v_model[words], axis=0)
    return torch.tensor(mean_vec.reshape(1, len(mean_vec)), dtype=torch.float)


class Calculator(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        ob.setup_s_ob_neg_aux_verb(self.matcher)
        ob.setup_s_ob_verb_error(self.matcher)
        ob.setup_s_ob_neg_verb(self.matcher)
        ob.setup_s_ob_but(self.matcher)
        ob.setup_s_ob_cond_pos(self.matcher)

    def utility(self, answers, posts):
        assert len(answers) == len(posts)
        util = 0
        for i in range(0, len(answers)):
            answer = answers[i]
            post = posts[i]
            answer_ob = self._get_ob(answer)
            if len(answer_ob) == 0:
                return 0

            post_ob = self._get_ob(post)
            diff = self._get_diff(post_ob.splitlines(keepends=True), answer_ob.splitlines(keepends=True))
            # it doesnt make sense to put softmax on one number
            util += len(diff.split()) / float(len(answer_ob.split()))

        return util / float(len(answers))

    def _get_diff(self, text_a, text_b):
        differ = Differ()
        diff_lines = differ.compare(text_a, text_b)
        diff = ' '.join([diff for diff in diff_lines if diff.startswith('+ ')]).replace('+ ', ' ')
        return diff

    def _get_ob(self, text):
        ob_str = ''
        for sentence in self.nlp(text).sents:
            sent = sentence.text.strip()
            if sent.startswith('>') or sent.endswith('?'):
                continue
            else:
                sent_nlp = self.nlp(sent)
                matches = self.matcher(sent_nlp)
                if len(matches) >= 1:
                    ob_str += sent + '\n'
        return ob_str


def evpi(vector_fpath, post_tsv, qa_tsv, n_epochs):
    glove2word2vec(vector_fpath, '../../embeddings_damevski/w2v_vectors.txt')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../../embeddings_damevski/w2v_vectors.txt')
    net = EvpiModel(w2v_model.vectors)
    calculator = Calculator()

    device = get_device()
    net.to(device)
    print('Running on {0}'.format(device))

    train_loader, test_loader = get_datasets(post_tsv, qa_tsv, w2v_model.vocab, batch_size=1)

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
