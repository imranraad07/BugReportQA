import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from data_generation import preprocessing as pp
import calculator as calc


def get_datasets(post_tsv, qa_tsv, word2index, train_ids, test_ids, batch_size, max_post_len, max_q_len, max_a_len,
                 shuffle=True):
    train_dataset = GithubDataset(post_tsv, qa_tsv, word2index, ids=train_ids, max_post_len=max_post_len,
                                  max_q_len=max_q_len, max_a_len=max_a_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    test_dataset = GithubDataset(post_tsv, qa_tsv, word2index, ids=test_ids, max_post_len=max_post_len,
                                 max_q_len=max_q_len, max_a_len=max_a_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return train_loader, test_loader


class GithubDataset(Dataset):

    def __init__(self, post_tsv, qa_tsv, word2index, ids, max_post_len, max_q_len, max_a_len):
        self.word2index = word2index
        self.max_post_len = max_post_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.dataset = self._build_dataset(post_tsv, qa_tsv, ids)

    def _build_dataset(self, post_tsv, qa_tsv, ids_file):
        ids = self._read_ids(ids_file)
        calculator = calc.Calculator()
        posts = pd.read_csv(post_tsv, sep='\t')
        qa = pd.read_csv(qa_tsv, sep='\t')
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
            if postid not in ids:
                continue

            post = row['title'] + ' ' + row['post']
            for i in range(1, 11):
                question = qa.iloc[idx]['q' + str(i)]
                answer = qa.iloc[idx]['a' + str(i)]
                utility = calculator.utility(answer, post)

                data = self._add_values(data, i, postid, post, question, answer, utility)

        dataset = pd.DataFrame(data)
        dataset = self._preprocess(dataset)

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset.iloc[idx]
        sample = self._to_dict(sample)
        return sample

    def _read_ids(self, ids_file):
        ids = set()
        with open(ids_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line):
                    ids.add(line)
        return ids

    def _to_dict(self, sample):
        new_sample = {'postid': sample['postid'],
                      'post': sample['post'],
                      'post_len': sample['post_len'],
                      'question': sample['question'],
                      'q_len': sample['q_len'],
                      'a_len': sample['a_len'],
                      'answer': sample['answer'],
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

    def _preprocess(self, dataset):
        dataset['post'] = dataset.post_origin.apply(self._clear_text)
        dataset['post'] = dataset.post.apply(self._word2index)
        dataset['post_len'] = dataset.post.apply(lambda x: self.max_post_len if len(x) > self.max_post_len else len(x))
        dataset['post'] = dataset.post.apply(self._padding_post)
        dataset['post'] = dataset.post.apply(self._to_tensor)
        dataset['post_len'] = dataset.post_len.apply(self._to_tensor)

        dataset['answer'] = dataset.answer_origin.apply(self._clear_text)
        dataset['answer'] = dataset.answer.apply(self._word2index)
        dataset['a_len'] = dataset.answer.apply(lambda x: self.max_a_len if len(x) > self.max_a_len else len(x))
        dataset['answer'] = dataset.answer.apply(self._padding_answer)
        dataset['answer'] = dataset.answer.apply(self._to_tensor)
        dataset['a_len'] = dataset.a_len.apply(self._to_tensor)

        dataset['question'] = dataset.question_origin.apply(self._clear_text)
        dataset['question'] = dataset.question.apply(self._word2index)
        dataset['q_len'] = dataset.question.apply(lambda x: self.max_q_len if len(x) > self.max_q_len else len(x))
        dataset['question'] = dataset.question.apply(self._padding_question)
        dataset['question'] = dataset.question.apply(self._to_tensor)
        dataset['q_len'] = dataset.q_len.apply(self._to_tensor)

        return dataset

    # all data transformations
    def _clear_text(self, text):
        return pp.clear_text(text, keep_punctuation=False)

    def _word2index(self, text):
        return [self.word2index[w].index for w in text.split() if w in self.word2index]

    def _to_tensor(self, value):
        return torch.tensor(value, dtype=torch.long)

    def _padding_post(self, values):
        return self._padding(values, self.max_post_len)

    def _padding_answer(self, values):
        return self._padding(values, self.max_a_len)

    def _padding_question(self, values):
        return self._padding(values, self.max_q_len)

    def _padding(self, values, limit):
        padded = np.zeros((limit,), dtype=np.int64)
        if len(values) > limit:
            padded[:] = values[:limit]
        else:
            padded[:len(values)] = values
        return padded
