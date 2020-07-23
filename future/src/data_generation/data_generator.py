import argparse
import csv
import os
import sys

sys.path.append(os.path.abspath('pattern_classification'))
from preprocessing import clear_text

from calculator import compute_utilities


def get_similar_docs(lucene_similar_docs):
    lucene_similar_docs_file = open(lucene_similar_docs, 'r')
    similar_docs = {}
    for line in lucene_similar_docs_file.readlines():
        parts = line.split()
        if len(parts) > 1:
            similar_docs[parts[0]] = parts[1:]
        else:
            print('Skip {0}. No similar posts found.'.format(parts[0]))
            # similar_docs[parts[0]] = []
    return similar_docs


def generate_docs_for_lucene_github(titles, posts, output_dir, post_ids):
    for postId in range(0, len(posts)):
        f = open(os.path.join(output_dir, post_ids[postId] + '.txt'), 'w')
        content = titles[post_ids[postId]] + ' ' + posts[post_ids[postId]]
        preprocessed_content = clear_text(content)
        f.write(preprocessed_content)
        f.close()


def create_tsv_files_github(post_data_tsv, qa_data_tsv, utility_data_tsv, post_titles, post_texts, post_questions,
                            post_answers, lucene_similar_posts):
    lucene_similar_posts = get_similar_docs(lucene_similar_posts)
    similar_posts = {}
    for line in lucene_similar_posts:
        if len(lucene_similar_posts[line]) < 11:
            continue
        postId = line
        similar_posts[postId] = lucene_similar_posts[line][1:11]

    with open(post_data_tsv, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['postid', 'title', 'post'])

    with open(qa_data_tsv, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['postid', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                             'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])
    qa_post_ids = {}
    for postId in similar_posts:
        with open(post_data_tsv, 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            if str(post_texts[postId]) == '':
                continue
            tsv_writer.writerow(
                [postId, post_titles[postId], post_texts[postId]])

            row_val = []
            row_val.append(postId)
            row_val.append(post_questions[postId])
            for i in range(9):
                row_val.append(post_questions[similar_posts[postId][i]])

            qa_ids = []
            qa_ids.append(postId)
            row_val.append(post_answers[postId])
            for i in range(9):
                row_val.append(post_answers[similar_posts[postId][i]])
                qa_ids.append(similar_posts[postId][i])
            qa_post_ids[postId] = qa_ids
            with open(qa_data_tsv, 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(row_val)

    compute_utilities(post_data_tsv, qa_data_tsv, qa_post_ids, utility_data_tsv)


def main(args):
    print('Parsing github posts...')
    csv.field_size_limit(sys.maxsize)

    repo_labels = {}
    with open(args.repo_label_csv) as csv_data_file:
        csv_reader = csv.reader((line.replace('\0', '') for line in csv_data_file))
        next(csv_reader)
        for row in csv_reader:
            repo_labels[row[0]] = row[1]

    issue_labels = {}
    with open(args.issue_label_csv) as csv_data_file:
        csv_reader = csv.reader((line.replace('\0', '') for line in csv_data_file))
        next(csv_reader)
        for row in csv_reader:
            issue_labels[row[0]] = row[1]

    issue_titles = {}
    with open(args.issue_title_csv) as csv_data_file:
        csv_reader = csv.reader((line.replace('\0', '') for line in csv_data_file))
        next(csv_reader)
        for row in csv_reader:
            issue_titles[row[0]] = row[1]

    post_ids = []
    post_titles = {}
    post_texts = {}
    post_questions = {}
    post_answers = {}
    idx = 0
    with open(args.github_csv) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        next(csvReader)
        for row in csvReader:
            post_ids.append(row[2])
            it = issue_titles[row[2]]
            post_titles[row[2]] = it
            rl = '' if row[0] not in repo_labels else repo_labels[row[0]]
            il = '' if row[2] not in issue_labels else issue_labels[row[2]]
            post_texts[row[2]] = row[3] + ' ' + rl + ' ' + il
            post_questions[row[2]] = (row[4])
            post_answers[row[2]] = (row[5])
            idx = idx + 1
    # print idx
    generate_docs_for_lucene_github(post_titles, post_texts, args.lucene_docs_dir, post_ids)
    os.system('cd %s && sh run_lucene.sh %s' % (args.lucene_dir, os.path.dirname(args.post_data_tsv)))

    create_tsv_files_github(args.post_data_tsv, args.qa_data_tsv, args.utility_data_tsv, post_titles, post_texts,
                            post_questions, post_answers, args.lucene_similar_posts)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--lucene_dir", type=str)
    argparser.add_argument("--lucene_docs_dir", type=str)
    argparser.add_argument("--lucene_similar_posts", type=str)
    argparser.add_argument("--word_embeddings", type=str)
    argparser.add_argument("--vocab", type=str)
    argparser.add_argument("--no_of_candidates", type=int, default=10)
    argparser.add_argument("--post_data_tsv", type=str)
    argparser.add_argument("--qa_data_tsv", type=str)
    argparser.add_argument("--utility_data_tsv", type=str)
    argparser.add_argument("--github_csv", type=str)
    argparser.add_argument("--issue_title_csv", type=str)
    argparser.add_argument("--repo_label_csv", type=str)
    argparser.add_argument("--issue_label_csv", type=str)
    args = argparser.parse_args()
    print(args)
    main(args)
