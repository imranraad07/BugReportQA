import csv
import json
import time
from datetime import datetime, timedelta

import requests
from nltk import sent_tokenize

from utils import mkdir


# import nltk
# nltk.download('punkt')

def get_comments(url, auth):
    response = requests.get(url, auth=auth)
    max_try = 60
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


def get_issues(repo, auth):
    url = "https://api.github.com/repos/{repo}/issues?state=all"
    url = url.format(repo=repo)
    return _getter(url, auth)


def _getter(url, auth):
    link = dict(next=url)
    print(link, time.ctime())
    while 'next' in link:
        response = requests.get(link['next'], auth=auth)
        # print(link, " ", response.status_code)

        max_try = 60
        # And.. if we didn't get good results, just bail.
        while response.status_code != 200:
            # return if 404
            if response.status_code == 404:
                print("Issues, Bad response code:", response.status_code, "returning....", time.ctime())
                for result in response.json():
                    yield result
            else:
                if max_try < 0:
                    break
                max_try = max_try - 1
                print("Issues, Bad response code:", response.status_code, "sleeping for 3 minutes....", time.ctime())
                time.sleep(180)
                # print("trying again....")
                response = requests.get(link['next'], auth=auth)

        if response.status_code == 200:
            for result in response.json():
                yield result
        else:
            return None

        link = _link_field_to_dict(response.headers.get('link', None))


def _link_field_to_dict(field):
    if not field:
        return dict()

    return dict([
        (
            part.split('; ')[1][5:-1],
            part.split('; ')[0][1:-1],
        ) for part in field.split(', ')
    ])


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


def read_github_issues(result_folder, result_file, auth):
    mkdir(result_folder)

    file = open("github_repos.txt", "r")
    github_repos = file.read().split()
    # print(github_repos)
    file.close()

    total_issues = 0
    repo_count = 0
    comment_added_csv_count = 0
    for repo in github_repos:
        issue_count = 0
        issues_this_repo = 0
        question_this_repo = 0
        try:
            for issue_data in get_issues(repo, auth):
                # github v3 api considers pull requests as issues. so filter them
                if 'pull_request' in issue_data:
                    continue
                if 'comments' not in issue_data:
                    print("comments is not in issue data")
                    continue
                # check if comment count is at least two
                if issue_data['comments'] < 2:
                    continue
                    # break

                # print(issue_count, " ", issue_data['title'])
                issue_count = issue_count + 1
                total_issues = total_issues + 1
                issues_this_repo = issues_this_repo + 1
                # print(total_issues)
                # # consider at most 1000 issues for each repo
                # if issue_count > 1500:
                #     break

                # issue_data = json.loads(issue)

                # label bug check
                is_label_bug = is_issue_label_bug(issue_data)
                if not is_label_bug:
                    continue

                comment_count = 0
                after_question = 0
                is_follow_up_question = False
                follow_up_question = ''
                follow_up_question_reply = ''
                if 'comments_url' not in issue_data:
                    print("comments_url is not in issue data")
                    continue
                comments = get_comments(issue_data['comments_url'], auth)
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
                        # just filtering by character count
                        comment_array = comment['body'].split()
                        if len(comment_array) > 30 or len(comment['body']) > 500:
                            continue
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
                            break

                if is_follow_up_question:
                    # print(follow_up_question, " ", comment['body'])
                    comment_added_csv_count = comment_added_csv_count + 1
                    question_this_repo = question_this_repo + 1
                    sw = csv.writer(open('{0}/{1}'.format(result_folder, result_file), 'a'))
                    column_data = issue_data['title']
                    if issue_data['body'] is not None:
                        column_data = column_data + "\n\n" + issue_data['body']
                    sw.writerow([
                        '{0}'.format(repo),
                        '{0}'.format(issue_data['html_url']),
                        '{0}'.format(column_data),
                        '{0}'.format(follow_up_question),
                        '{0}'.format(follow_up_question_reply)
                    ])
        except ConnectionError as e:
            print("ConnectionError for repo", repo, time.ctime(), str(e))
            with open("exception_repos.txt", "a") as exf:
                exf.write(repo + "\n")
        except Exception as e:
            print("exception for repo", repo, time.ctime(), str(e))
            with open("exception_repos.txt", "a") as exf:
                exf.write(repo + "\n")
        repo_count = repo_count + 1
        print(repo_count, total_issues, comment_added_csv_count, " :::: ", issues_this_repo, question_this_repo)


if __name__ == '__main__':
    with open('credentials.json') as json_file:
        data = json.load(json_file)
    username = data['username']
    password = data['password']
    auth = (username, password)
    read_github_issues('results', 'github_data.csv', auth)
