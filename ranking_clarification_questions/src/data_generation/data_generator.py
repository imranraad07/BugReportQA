import argparse
import csv
import os

from parse import *
from post_ques_ans_generator import *


def get_similar_docs(lucene_similar_docs):
    lucene_similar_docs_file = open(lucene_similar_docs, 'r')
    similar_docs = {}
    for line in lucene_similar_docs_file.readlines():
        parts = line.split()
        if len(parts) > 1:
            similar_docs[parts[0]] = parts[1:]
        else:
            similar_docs[parts[0]] = []
    return similar_docs


def generate_docs_for_lucene(post_ques_answers, posts, output_dir):
    for postId in post_ques_answers:
        f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
        content = ' '.join(posts[postId].title).encode('utf-8') + ' ' + ' '.join(posts[postId].body).encode('utf-8')
        f.write(content)
        f.close()


def generate_docs_for_lucene_github(titles, posts, output_dir):
    for postId in range(0, len(posts)):
        f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
        # content = titles[postId] + ' ' + posts[postId]
        # content = content.decode('utf-8', errors='ignore').encode('utf-8')
        content = ' '.join(titles[postId].decode('utf-8', errors='ignore').encode('utf-8')) + ' ' + ' '.join(
            posts[postId].decode('utf-8', errors='ignore').encode('utf-8'))
        f.write(content)
        f.close()


def create_tsv_files(post_data_tsv, qa_data_tsv, post_ques_answers, lucene_similar_posts):
    lucene_similar_posts = get_similar_docs(lucene_similar_posts)
    similar_posts = {}
    for line in lucene_similar_posts.readlines():
        splits = line.strip('\n').split()
        if len(splits) < 11:
            continue
        postId = splits[0]
        similar_posts[postId] = splits[1:11]
    post_data_tsv_file = open(post_data_tsv, 'w')
    post_data_tsv_file.write('postid\ttitle\tpost\n')
    qa_data_tsv_file = open(qa_data_tsv, 'w')
    qa_data_tsv_file.write('postid\tq1\tq2\tq3\tq4\tq5\tq6\tq7\tq8\tq9\tq10\ta1\ta2\ta3\ta4\ta5\ta6\ta7\ta8\ta9\ta10\n')
    print len(similar_posts)
    for postId in similar_posts:
        post_data_tsv_file.write('%s\t%s\t%s\n' % (postId, \
                                                   ' '.join(post_ques_answers[postId].post_title), \
                                                   ' '.join(post_ques_answers[postId].post)))
        line = postId
        for i in range(10):
            line += '\t%s' % ' '.join(post_ques_answers[similar_posts[postId][i]].question_comment)
        for i in range(10):
            line += '\t%s' % ' '.join(post_ques_answers[similar_posts[postId][i]].answer)
        line += '\n'
        qa_data_tsv_file.write(line)


def create_tsv_files_github(post_data_tsv, qa_data_tsv, post_ids, post_titles, post_texts, post_questions,
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
        tsv_writer.writerow(['postid', 'title', 'post', 'issueid'])

    with open(qa_data_tsv, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['postid', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
                             'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])

    for postId in similar_posts:
        post_id = int(postId)
        with open(post_data_tsv, 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(
                [post_id, post_titles[post_id].decode('utf-8', errors='ignore').encode('utf-8'),
                 post_texts[post_id].decode('utf-8', errors='ignore').encode('utf-8'),
                 post_ids[post_id].decode('utf-8', errors='ignore').encode('utf-8')])

        row_val = []
        row_val.append(postId)
        for i in range(10):
            row_val.append(
                post_questions[int(similar_posts[postId][i])].decode('utf-8', errors='ignore').encode('utf-8'))
        for i in range(10):
            row_val.append(post_answers[int(similar_posts[postId][i])].decode('utf-8', errors='ignore').encode('utf-8'))
        with open(qa_data_tsv, 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(row_val)

    with open('../../data/github/test_ids', 'w') as out_file:
        idx = 0
        for postId in similar_posts:
            if idx < 1000:
                out_file.write(postId + "\n")
            idx = idx + 1
    with open('../../data/github/tune_ids', 'w') as out_file:
        idx = 1000
        for postId in similar_posts:
            if idx >= 1000 and idx < 2000:
                out_file.write(postId + "\n")
            idx = idx + 1
    with open('../../data/github/train_ids', 'w') as out_file:
        idx = 2000
        for postId in similar_posts:
            if idx >= 2000:
                out_file.write(postId + "\n")
            idx = idx + 1


def main(args):
    if args.site_name == "github":
        print 'Parsing github posts...'
        csv.field_size_limit(sys.maxsize)
        post_ids = []
        post_titles = []
        post_texts = []
        post_questions = []
        post_answers = []
        idx = 0
        with open(args.github_csv) as csvDataFile:
            csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
            for row in csvReader:
                post_ids.append(row[2].decode('utf-8', errors='ignore').encode('utf-8'))
                post_titles.append(row[3].partition('\n')[0].decode('utf-8', errors='ignore').encode('utf-8'))
                post_texts.append(row[3].partition('\n')[-1].decode('utf-8', errors='ignore').encode('utf-8'))
                post_questions.append(row[4].decode('utf-8', errors='ignore').encode('utf-8'))
                post_answers.append(row[5].decode('utf-8', errors='ignore').encode('utf-8'))
                idx = idx + 1
        # print idx
        generate_docs_for_lucene_github(post_titles, post_texts, args.lucene_docs_dir)
        os.system('cd %s && sh run_lucene.sh %s' % (args.lucene_dir, os.path.dirname(args.post_data_tsv)))

        create_tsv_files_github(args.post_data_tsv, args.qa_data_tsv, post_ids, post_titles, post_texts, post_questions,
                                post_answers, args.lucene_similar_posts)

    else:
        start_time = time.time()
        print 'Parsing posts...'
        post_parser = PostParser(args.posts_xml)
        post_parser.parse()
        posts = post_parser.get_posts()
        print 'Size: ', len(posts)
        print 'Done! Time taken ', time.time() - start_time
        print

        start_time = time.time()
        print 'Parsing posthistories...'
        posthistory_parser = PostHistoryParser(args.posthistory_xml)
        posthistory_parser.parse()
        posthistories = posthistory_parser.get_posthistories()
        print 'Size: ', len(posthistories)
        print 'Done! Time taken ', time.time() - start_time
        print

        start_time = time.time()
        print 'Parsing question comments...'
        comment_parser = CommentParser(args.comments_xml)
        comment_parser.parse_all_comments()
        question_comments = comment_parser.get_question_comments()
        all_comments = comment_parser.get_all_comments()
        print 'Size: ', len(question_comments)
        print 'Done! Time taken ', time.time() - start_time
        print

        start_time = time.time()
        print 'Loading vocab'
        vocab = p.load(open(args.vocab, 'rb'))
        print 'Done! Time taken ', time.time() - start_time
        print

        start_time = time.time()
        print 'Loading word_embeddings'
        word_embeddings = p.load(open(args.word_embeddings, 'rb'))
        word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
        print 'Done! Time taken ', time.time() - start_time
        print

        start_time = time.time()
        print 'Generating post_ques_ans...'
        post_ques_ans_generator = PostQuesAnsGenerator()
        post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, all_comments, posthistories,
                                                             vocab,
                                                             word_embeddings)
        print 'Size: ', len(post_ques_answers)
        print 'Done! Time taken ', time.time() - start_time
        print

        generate_docs_for_lucene(post_ques_answers, posts, args.lucene_docs_dir)
        os.system('cd %s && sh run_lucene.sh %s' % (args.lucene_dir, os.path.dirname(args.post_data_tsv)))

        create_tsv_files(args.post_data_tsv, args.qa_data_tsv, post_ques_answers, args.lucene_similar_posts)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--posts_xml", type=str)
    argparser.add_argument("--comments_xml", type=str)
    argparser.add_argument("--posthistory_xml", type=str)
    argparser.add_argument("--lucene_dir", type=str)
    argparser.add_argument("--lucene_docs_dir", type=str)
    argparser.add_argument("--lucene_similar_posts", type=str)
    argparser.add_argument("--word_embeddings", type=str)
    argparser.add_argument("--vocab", type=str)
    argparser.add_argument("--no_of_candidates", type=int, default=10)
    argparser.add_argument("--site_name", type=str)
    argparser.add_argument("--post_data_tsv", type=str)
    argparser.add_argument("--qa_data_tsv", type=str)
    argparser.add_argument("--github_csv", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)
