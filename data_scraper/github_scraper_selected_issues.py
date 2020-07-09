import argparse
import csv
import sys
from datetime import datetime, timedelta

from github_apis_v3 import *
from nltk import sent_tokenize
from queries import *
from utils import question_identifier

# import nltk
# nltk.download('punkt')


headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e',
           'Accept': 'application/vnd.github.mercy-preview+json'}


def read_github_issues(issue_id):
    try:
        url = "https://api.github.com/repos/{repo_issue_id}"
        url = url.format(repo_issue_id=issue_id)
        print(url)
        issue_data = get_an_issue(url, headers)
        print(issue_data)

        if issue_data is None:
            return None

        # check if comment count is at least two
        if issue_data['comments'] < 2:
            return None

        comment_count = 0
        after_question = 0
        is_follow_up_question = False
        is_follow_up_question_answer = False
        follow_up_question = ''
        follow_up_question_reply = ''
        if 'comments_url' not in issue_data:
            print("comments_url is not in issue data")
            return None
        comments = get_comments(issue_data['comments_url'], headers)
        if comments is None:
            return None

        # reading the comments
        for comment in comments:
            if 'user' not in comment:
                print("user is not in comment data")
                continue
            if 'user' not in issue_data:
                print("user is not in issue data")
                continue

            # comment within 60 days of issue creation
            d1 = datetime.strptime(comment['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            d2 = datetime.strptime(issue_data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            if d1 - d2 > timedelta(days=60):
                # print(d1-d2)
                continue

            if not is_follow_up_question and comment_count < 3:
                comment_count = comment_count + 1
                # if comment author and issue author are same, then discard the comment
                if comment['user']['id'] == issue_data['user']['id']:
                    continue
                follow_up_question = comment['body']
                for sentence in sent_tokenize(comment['body']):
                    sentence = sentence.strip()
                    if sentence.startswith(">"):
                        continue
                    elif question_identifier(sentence):
                        # if sentence starts with @someone, check if this @someone is original issue author or not
                        if sentence.startswith("@"):
                            mentioned_login = sentence.split()[0]
                            github_login = "@{0}".format(issue_data['user']['login'])
                            if mentioned_login != github_login:
                                break
                        is_follow_up_question = True
                        idx = comment['body'].find(sentence)
                        follow_up_question = comment['body'][idx:]
                        after_question = 0
                        break
            elif is_follow_up_question and after_question < 3:
                after_question = after_question + 1
                if issue_data['user']['login'] == comment['user']['login']:
                    follow_up_question_reply = comment['body']
                    is_follow_up_question_answer = True
                    break

        if is_follow_up_question and is_follow_up_question_answer:
            # filter by word count and #characters
            comment_array = follow_up_question.split()
            if len(comment_array) > 30 or len(follow_up_question) > 300:
                return None

            if issue_data['body'] is None:
                return None
            original_post = issue_data['body']
            return original_post, follow_up_question, follow_up_question_reply
    except Exception as ex:
        print("exception on issue, ", issue_id, str(ex))
    return None


def parse_repos(input_file, result_file):
    csv_file = open(result_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])

    with open(input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            issue_data = read_github_issues(row[1][19:])
            if issue_data is not None:
                csv_writer.writerow([row[0], row[1], row[2], issue_data[0], issue_data[1], issue_data[2]])
    csv_file.close()


def main(args):
    parse_repos(args.input_file, args.output_file)


if __name__ == '__main__':
    print(sys.argv)

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str,
                           default='../data/bug_reports/github_data_2008.csv')
    argparser.add_argument("--output_file", type=str,
                           default='../data/bug_reports/output_dataset.csv')

    args = argparser.parse_args()

    csv.field_size_limit(sys.maxsize)
    print(args)
    print("")
    main(args)
