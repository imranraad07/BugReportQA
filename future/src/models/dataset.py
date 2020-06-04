import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import preprocessing as pp
import models.calculator as calc


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


class GithubDataset(Dataset):

    def __init__(self, post_tsv, qa_tsv, train, transform=None):
        self.transform = transform
        self.dataset = self._build_dataset(post_tsv, qa_tsv, train)

    def _build_dataset(self, post_tsv, qa_tsv, train):
        calculator = calc.Calculator()
        posts = pd.read_csv(post_tsv, sep='\t')[:13]
        qa = pd.read_csv(qa_tsv, sep='\t')[:13]
        data = {'postid': list(),
                'post_origin': list(),
                'question_origin': list(),
                'answer_origin': list(),
                'label': list(),
                'utility': list()}

        for idx, row in posts.iterrows():
            # TODO: this should be covered when building dataset
            if str(row['post']) == 'nan':
                continue
            postid = row['postid']
            post = row['title'] + ' ' + row['post']
            for i in range(1, 11):
                question = qa.iloc[idx]['q' + str(i)]
                answer = qa.iloc[idx]['a' + str(i)]
                utility = calculator.utility(answer, post)

                data = self._add_values(data, i, postid, post, question, answer, utility)

        dataset = pd.DataFrame(data)

        instances_no = len(dataset) / 10
        train_instances = int(instances_no * 0.9)
        if train is True:
            dataset = dataset[:(train_instances * 10)]
        else:
            dataset = dataset[(train_instances * 10):].reset_index(drop=True)
        return dataset

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
        new_sample = {'postid': sample['postid'],
                      'post_origin': sample['post_origin'],
                      'question_origin': sample['question_origin'],
                      'answer_origin': sample['answer_origin'],
                      'utility': sample['utility'],
                      'label': sample['label']}
        return new_sample

    def _add_values(self, data, index, postid, post, question, answer, utility):
        data['postid'].append(postid)
        data['post_origin'].append(post)
        data['answer_origin'].append(answer)
        data['question_origin'].append(question)
        data['utility'].append(utility)
        if index == 1:
            data['label'].append(1)
        else:
            data['label'].append(0)
        return data


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
        return sample
