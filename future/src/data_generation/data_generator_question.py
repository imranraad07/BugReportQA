import argparse
import csv
import os
import sys


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


def create_tsv_files_github_question(dataset_csv, data_avg_file, post_questions, post_answers, lucene_similar_posts):
    lucene_similar_posts = get_similar_docs(lucene_similar_posts)
    similar_posts = {}
    for line in lucene_similar_posts:
        if len(lucene_similar_posts[line]) < 11:
            continue
        postId = line
        similar_posts[postId] = lucene_similar_posts[line][1:11]

    with open(data_avg_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['postid', 'postid1', 'q1', 'a1', 'postid2', 'q2', 'a2',
                             'postid3', 'q3', 'a3', 'postid4', 'q4', 'a4', 'postid5', 'q5', 'a5', 'postid6', 'q6', 'a6',
                             'postid7', 'q7', 'a7', 'postid8', 'q8', 'a8', 'postid9', 'q9', 'a9', 'postid10', 'q10',
                             'a10'])

    print(similar_posts['ron190_jsql-injection_issues_4062'])

    with open(dataset_csv) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        next(csvReader)
        for row in csvReader:
            postId = row[2]
            if postId in similar_posts:
                row_val = []
                row_val.append(postId)

                row_val.append(postId)
                row_val.append(post_questions[postId])
                row_val.append(post_answers[postId])

                for i in range(9):
                    row_val.append(similar_posts[postId][i])
                    row_val.append(post_questions[similar_posts[postId][i]])
                    row_val.append(post_answers[similar_posts[postId][i]])

                # print(row_val)
                with open(data_avg_file, 'a') as out_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    tsv_writer.writerow(row_val)


def generate_docs_for_lucene_github_questions(type, post_questions, post_answers, output_dir, post_ids):
    for postId in range(0, len(post_questions)):
        f = open(os.path.join(output_dir, post_ids[postId] + '.txt'), 'w')
        if 'answer' in type:
            content = post_questions[post_ids[postId]] + " " + post_answers[post_ids[postId]]
        else:
            content = post_questions[post_ids[postId]]
        f.write(content)
        f.close()


def main(args):
    print('Parsing github posts...')
    csv.field_size_limit(sys.maxsize)

    post_ids = []
    post_questions = {}
    post_answers = {}
    idx = 0
    with open(args.github_csv) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        next(csvReader)
        for row in csvReader:
            post_ids.append(row[2])
            post_questions[row[2]] = (row[4])
            post_answers[row[2]] = (row[5])
            idx = idx + 1
    # print idx
    generate_docs_for_lucene_github_questions(args.type, post_questions, post_answers, args.lucene_docs_dir, post_ids)
    os.system('cd %s && sh run_lucene.sh' % (args.lucene_dir))
    create_tsv_files_github_question(args.github_csv, args.data_avg_file, post_questions, post_answers,
                                     args.lucene_similar_posts)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--type", type=str)
    argparser.add_argument("--lucene_dir", type=str)
    argparser.add_argument("--lucene_docs_dir", type=str)
    argparser.add_argument("--lucene_similar_posts", type=str)
    argparser.add_argument("--data_avg_file", type=str)
    argparser.add_argument("--github_csv", type=str)
    args = argparser.parse_args()
    print("args are...")
    print(args.type)
    print(args)
    main(args)
