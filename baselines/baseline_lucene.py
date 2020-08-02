import argparse
import pandas as pd
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-tsv', help='File path to post_tsv produced by Lucene',
                        default='data/datasets_final_tag/post_data.tsv', )
    parser.add_argument('--qa-tsv', help='File path to qa_tsv produced by Lucene',
                        default='data/datasets_final_tag/qa_data.tsv', )
    parser.add_argument('--test-ids', help='File path to test ids',
                        default='data/datasets_final_tag/test_ids.txt', )
    parser.add_argument('--output-ranking-file', help='Output file to save ranking',
                        default='../results/datasets_final_tag/ranking_baseline_lucene.csv')
    return parser.parse_args()


def run():
    args = parse_args()
    post_data = pd.read_csv(args.post_tsv, sep='\t')
    qa_data = pd.read_csv(args.qa_tsv, sep='\t')
    with open(args.test_ids) as f:
        ids = set([x.strip() for x in f.readlines()])
    ranking = lucene_ranking(post_data, qa_data, ids)
    save_ranking(args.output_ranking_file, ranking)


def lucene_ranking(post_data, qa_data, ids):
    dataset = dict()
    print("calculating lucene ranking...")
    for idx, row in post_data.iterrows():
        postid = row['postid']
        if postid in ids:
            dataset[postid] = (row['post'], list(), list())
            correct_question = qa_data.iloc[idx]['q1']
            correct_answer = qa_data.iloc[idx]['a1']
            dataset[postid][2].append(correct_question)
            dataset[postid][2].append(correct_answer)

            for i in range(2, 11):
                question = qa_data.iloc[idx]['q' + str(i)]
                answer = qa_data.iloc[idx]['a' + str(i)]
                score = i
                # print(postid, score)
                dataset[postid][1].append((score, question, answer))

            # putting the correct q&a to the back of the list
            dataset[postid][1].append((11, correct_question, correct_answer))
    print("done")
    return dataset


def save_ranking(output_file, results):
    print("saving ranking...")
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

            values = sorted(values, key=lambda x: x[0])
            for score, question, answer in values:
                record.append(question)
                record.append(answer)
            writer.writerow(record)
    print("done")


if __name__ == '__main__':
    run()
