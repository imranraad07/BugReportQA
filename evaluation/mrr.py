import pandas as pd


def compute(data_fpath):
    df = pd.read_csv(data_fpath, sep=',', index_col=False)

    ranks = list()
    for index, row in df.iterrows():
        correct = row['correct_question']
        for i in range(1, 11):
            q = row['q' + str(i)]
            if q == correct:
                ranks.append(1.0 / float(i))
                break
            if i == 10:
                raise ValueError('This shouldnt happen. Row is\n{0}'.format(row))

    mrr = sum(ranks) / float(len(ranks))
    return mrr
