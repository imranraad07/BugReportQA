import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import preprocessing as pp


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
