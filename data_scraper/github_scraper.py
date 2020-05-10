import csv
import json
import time
from datetime import datetime, timedelta

import requests
from nltk import sent_tokenize

from data_scraper.github_text_filter import filter_nontext
from data_scraper.queries import *

# import nltk
# nltk.download('punkt')

headers = {"Authorization": "Bearer a4a37bc57f01dfef13d3c5f629dbc51800d554ca"}
with open('../credentials.json') as json_file:
    data = json.load(json_file)
username = data['username']
password = data['password']
auth = (username, password)


def get_comments(url):
    response = requests.get(url, auth=auth)
    max_try = 20
    while response.status_code != 200:
        if max_try < 0:
            break
        max_try = max_try - 1
        print("Comments, Bad response code:", response.status_code, "sleeping for 3 minutes....", time.ctime())
        time.sleep(180)
        # print("trying again....")
        response = requests.get(url, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_an_issue(repo, issue_id):
    url = "https://api.github.com/repos/{repo}/issues/{issue_id}"
    url = url.format(repo=repo, issue_id=issue_id)

    response = requests.get(url, auth=auth)
    max_try = 20
    while response.status_code != 200:
        if max_try < 0:
            break
        max_try = max_try - 1
        print("Issues, Bad response code:", response.status_code, "sleeping for 3 minutes....", time.ctime())
        time.sleep(180)
        response = requests.get(url, auth=auth)

    if response.status_code == 200:
        _json = response.json()
        return _json
    else:
        return None


def question_identifier(sentence):
    start_words = ['who', 'what', 'when', 'where', 'why', 'which', 'how', "while", "do", "does", "did", "will", "would",
                   "can", "could", "shall", "should", "may", "might", "must"]
    flag = False
    for word in start_words:
        if sentence.lower().startswith(word.lower()):
            flag = True
            # return True
    if flag and sentence.endswith('?'):
        # print(sentence)
        return True
    return False


def is_issue_label_bug(issue_data):
    if 'labels' not in issue_data:
        print("labels is not in issue data")
        return False
    is_label_bug = False
    labels = issue_data['labels']
    if labels is None:
        return False
    for label_data in labels:
        if 'name' not in label_data:
            print("name is not in label data")
            return False
        label_text = label_data['name']
        label_desc = label_data['description']
        if label_desc is None:
            label_desc = ""
        if ("bug" in label_text) or ("defeat" in label_text) or ("bug" in label_desc):
            is_label_bug = True
            break

    if 'title' not in issue_data:
        print("title is not in issue data")
    else:
        if issue_data['title'] is not None and "bug" in issue_data['title']:
            is_label_bug = True
    return is_label_bug


def read_github_issues(github_repo, bug_ids, csv_writer):
    # csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])

    for issue_id in bug_ids:
        print("issue_id", issue_id)
        try:
            issue_data = get_an_issue(github_repo, issue_id)
            print(issue_data)
            if issue_data is None:
                continue
            # github v3 api considers pull requests as issues. so filter them
            if 'pull_request' in issue_data:
                continue
            if 'comments' not in issue_data:
                print("comments is not in issue data")
                continue
            # check if comment count is at least two
            if issue_data['comments'] < 2:
                continue

            comment_count = 0
            after_question = 0
            is_follow_up_question = False
            is_follow_up_question_answer = False
            follow_up_question = ''
            follow_up_question_reply = ''
            if 'comments_url' not in issue_data:
                print("comments_url is not in issue data")
                continue
            comments = get_comments(issue_data['comments_url'])
            if comments is None:
                continue

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
                                is_mentioned = sentence.split()[0]
                                github_login = "@{0}".format(issue_data['user']['login'])
                                if is_mentioned != github_login:
                                    break
                            is_follow_up_question = True
                            idx = comment['body'].find(sentence)
                            follow_up_question = comment['body'][idx:]
                            after_question = 0
                            break
                elif follow_up_question and after_question < 3:
                    after_question = after_question + 1
                    if issue_data['user']['login'] == comment['user']['login']:
                        follow_up_question_reply = comment['body']
                        is_follow_up_question_answer = True
                        break

            if is_follow_up_question and is_follow_up_question_answer:
                # just filtering by character count
                comment_array = follow_up_question.split()
                if len(comment_array) > 30 or len(follow_up_question) > 300:
                    continue

                original_post = issue_data['title']
                if issue_data['body'] is not None:
                    original_post = original_post + "\n\n" + filter_nontext(issue_data['body'])
                follow_up_question = filter_nontext(follow_up_question)
                follow_up_question_reply = filter_nontext(follow_up_question_reply)
                postid = issue_data['html_url'][19:]
                postid = postid.replace("/", "_")
                write_row = [github_repo, issue_data['html_url'], postid, original_post, follow_up_question,
                             follow_up_question_reply]
                csv_writer.writerow(write_row)
                print(write_row)
        except Exception as ex:
            print("exception on issue, ", issue_id, str(ex))


def get_edits(repo_url, issue_no):
    tokens = repo_url.split('/')
    owner = '\"' + tokens[3] + '\"'
    name = '\"' + tokens[4] + '\"'

    query = edit_query.substitute(owner=owner, name=name, number=issue_no)
    failed_cnt = 0
    while failed_cnt < 20:
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

        if request.status_code == 200:
            result = request.json()
            edits = [(x['node']['diff'], x['node']['createdAt']) for x in
                     result['data']['repository']['issue']['userContentEdits']['edges']]
            return edits
        elif request.status_code == 502:
            print("Query failed to run by returning code of 502 for repo {0} - issue {1}. Try again in 30s...".format(
                repo_url, issue_no))
            time.sleep(30)
            failed_cnt += 1
            continue
        elif request.status_code == 403:
            print('Abusive behaviour mechanism was triggered. Wait 3 min.')
            time.sleep(180)
            failed_cnt += 1
            continue
        else:
            raise Exception(
                "Query failed to run by returning code of {0}. Query params: url:{1}, issue:{2}.".format(
                    request.status_code, repo_url, issue_no))


def parse_repos(file_name, result_folder, result_file):
    csv_file = open('{0}/{1}'.format(result_folder, result_file), 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])

    with open(file_name) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csvReader))
        for row in csvReader:
            # print(row[1][19:], row[9:])
            read_github_issues(row[1][19:], row[9:], csv_writer)
        # print(count)
    csv_file.close()


if __name__ == '__main__':
    # read_github_issues('../data/github_repos/github_repos_name_sorted.txt', '../data/bug_reports',
    #                    'github_data.csv', auth)
    parse_repos('../data/repos/repos_final2009.csv', '../data/bug_reports', 'github_data_2009.csv')
