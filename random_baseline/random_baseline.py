import os
import sys

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('../pattern_classification'))

import argparse
import random
import pandas as pd
import csv
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-tsv', help='File path to post_tsv produced by Lucene', required=True)
    parser.add_argument('--qa-tsv', help='File path to qa_tsv produced by Lucene', required=True)
    parser.add_argument('--test-ids', help='File path to test ids', required=True)
    parser.add_argument('--output-ranking-file', help='Output file to save ranking', required=True)
    return parser.parse_args()


def run():
    args = parse_args()
    post_data = pd.read_csv(args.post_tsv, sep='\t')
    qa_data = pd.read_csv(args.qa_tsv, sep='\t')
    with open(args.test_ids) as f:
        ids = set([x.strip() for x in f.readlines()])

    ranking = random_ranking(post_data, qa_data, ids)
    save_ranking(args.output_ranking_file, ranking)


def random_ranking(post_data, qa_data, ids):
    dataset = dict()
    for idx, row in post_data.iterrows():
        postid = row['postid']
        if row['postid'] in ids:
            dataset[postid] = (postid, list(), list())
            correct_question = qa_data.iloc[idx]['q1']
            correct_answer = qa_data.iloc[idx]['a1']
            dataset[postid][2].append(correct_question)
            dataset[postid][2].append(correct_answer)

            for i in range(1, 11):
                question = qa_data.iloc[idx]['q' + str(i)]
                answer = qa_data.iloc[idx]['a' + str(i)]
                score = random.uniform(0, 1)
                dataset[postid][1].append((score, question, answer))
    return dataset


def save_ranking(output_file, results):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['issueid', 'post', 'correct_question', 'correct_a']
        for i in range(1, 11):
            header.append('q' + str(i))
            header.append('a' + str(i))
        writer.writerow(header)

        for postid in results:
            post, values, correct = results[postid]

            record = [postid, post, correct[0], correct[1]]

            values = sorted(values, key=lambda x: x[0], reverse=True)
            for score, question, answer in values:
                record.append(question)
                record.append(answer)

            writer.writerow(record)


if __name__ == '__main__':
    run()
