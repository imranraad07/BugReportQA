import csv
import json

import requests
from nltk import sent_tokenize
from nltk import word_tokenize
from datetime import datetime, timedelta

from utils import mkdir

# import nltk
# nltk.download('punkt')

github_repos = [
    "ReactiveX/RxNetty", "elastic/elasticsearch-hadoop", "springfox/springfox", "google/conscrypt",
    "google/dagger", "spring-projects/spring-security", "spring-projects/spring-session",

    "socketio/socket.io-client-java", "square/moshi", "scribejava/scribejava", "OpenRefine/OpenRefine",
    "square/javapoet", "mockito/mockito", "junit-team/junit4", "mybatis/mybatis-3", "bazelbuild/bazel",
    "orhanobut/logger", "realm/realm-java", "greenrobot/greenDAO", "google/ExoPlayer", "jfeinstein10/SlidingMenu",
    "android/plaid", "Tencent/tinker", "chrisbanes/PhotoView", "afollestad/material-dialogs",
    "Netflix/Hystrix", "libgdx/libgdx", "netty/netty", "facebook/fresco", "skylot/jadx", "square/picasso",
    "greenrobot/EventBus", "zxing/zxing", "square/leakcanary", "airbnb/lottie-android",
    "JakeWharton/butterknife", "spring-projects/spring-framework", "bumptech/glide", "PhilJay/MPAndroidChart",
    "google/guava", "spring-projects/spring-boot", "square/okhttp", "square/retrofit", "elastic/elasticsearch",
    "ReactiveX/RxJava",
]


def get_comments(url, auth):
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.json()


def get_issues(repo, auth):
    # url = "https://api.github.com/repos/{repo}/issues?state=closed&sort=comments-desc"
    url = "https://api.github.com/repos/{repo}/issues?state=closed"
    url = url.format(repo=repo)
    return _getter(url, auth)


def _getter(url, auth):
    link = dict(next=url)
    print(link)
    while 'next' in link:
        # print(link)
        response = requests.get(link['next'], auth=auth)
        # And.. if we didn't get good results, just bail.
        if response.status_code != 200:
            print("Bad response code: ", response.status_code)
        #     raise IOError(
        #         "Non-200 status code %r; %r; %r" % (
        #             response.status_code, url, response.json()))

        for result in response.json():
            yield result

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


def check(sentence):
    start_words = ['who', 'what', 'when', 'where', 'why', 'which', 'how']
    if sentence.endswith('?'):
        return True
    for word in start_words:
        if sentence.startswith(word):
            return True
    return False


def read_github_issues(result_folder, result_file, auth):
    mkdir(result_folder)
    total_issues = 0
    comment_added_csv_count = 0
    for repo in github_repos:
        issue_count = 0
        for issue_data in get_issues(repo, auth):
            # github v3 api considers pull requests as issues. so filter them
            if 'pull_request' in issue_data:
                continue
            # print(issue_count, " ", issue_data['title'])
            issue_count = issue_count + 1
            total_issues = total_issues + 1
            # print(total_issues)
            # consider at most 1000 issues for each repo
            if issue_count > 1000:
                break

            # issue_data = json.loads(issue)

            # label bug check
            is_label_bug = False

            if 'labels' not in issue_data:
                print("labels is not in issue data")
                continue

            labels = issue_data['labels']
            if labels is None:
                continue
            for label_data in labels:
                if 'name' not in label_data:
                    print("name is not in label data")
                    continue
                label_text = label_data['name']
                label_desc = label_data['description']
                if label_desc is None:
                    label_desc = ""
                if ("bug" in label_text) or ("bug" in label_desc):
                    is_label_bug = True
                    break

            if 'title' not in issue_data:
                print("title is not in issue data")
            else:
                if issue_data['title'] is not None and "bug" in issue_data['title']:
                    is_label_bug = True

            # if issue['body'] is not None and "bug" in issue['body']:
            #     is_label_bug = True

            if not is_label_bug:
                continue

            # print(comment_added_csv_count, " ", issue['comments'], " ", issue['labels'], " ", issue['comments_url'])
            comment_count = 0
            is_follow_up_question = False
            if 'comments' not in issue_data:
                print("comments is not in issue data")
                continue
            if 'comments_url' not in issue_data:
                print("comments_url is not in issue data")
                continue
            # check if comment count is at least two
            if issue_data['comments'] < 2:
                continue
            comments = get_comments(issue_data['comments_url'], auth)
            if comments is None:
                continue
            for comment in comments:
                comment_count = comment_count + 1
                # look up to 3 comments
                if comment_count > 3:
                    break

                if 'user' not in comment:
                    print("user is not in comment data")
                    continue
                if 'user' not in issue_data:
                    print("user is not in issue data")
                    continue

                d1 = datetime.strptime(comment['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                d2 = datetime.strptime(issue_data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                if d1 - d2 > timedelta(days=60):
                    # print(d1-d2)
                    continue

                # if comment author and issue author are same, then discard the comment
                if comment['user']['id'] == issue_data['user']['id']:
                    continue
                # just filtering by character count
                if len(comment['body']) > 300:
                    continue
                follow_up_question = comment['body']
                for sentence in sent_tokenize(comment['body']):
                    sentence = sentence.strip()
                    if sentence.startswith(">"):
                        continue
                    # elif is_follow_up_question:
                    #     follow_up_question = follow_up_question.join(sentence)
                    elif check(sentence):
                        # if sentence starts with @someone, check if this @someone is original issue author or not
                        if sentence.startswith("@"):
                            # print(sentence)
                            is_mentioned = sentence.split()[0]
                            github_login = "@{0}".format(issue_data['user']['login'])
                            # print(is_mentioned, " ", github_login, " ", is_mentioned == github_login)
                            if is_mentioned != github_login:
                                break
                        is_follow_up_question = True
                        idx = comment['body'].find(sentence)
                        # print(idx)
                        # if idx != 0:
                        follow_up_question = comment['body'][idx:]
                        # print(idx, " ", follow_up_question, " ", comment['body'])
                        break

                if is_follow_up_question:
                    # print(follow_up_question, " ", comment['body'])
                    comment_added_csv_count = comment_added_csv_count + 1
                    sw = csv.writer(open('{0}/{1}'.format(result_folder, result_file), 'a'))
                    sw.writerow([
                        '{0}'.format(repo),
                        '{0}'.format(issue_data['body']),
                        '{0}'.format(comment['html_url']),
                        '{0}'.format(follow_up_question)
                        # '{0}'.format(comment['body'])
                    ])
                    break
        print(total_issues, " ", comment_added_csv_count)


if __name__ == '__main__':
    with open('credentials.json') as json_file:
        data = json.load(json_file)
    username = data['username']
    password = data['password']
    auth = (username, password)
    read_github_issues('results', 'github_data.csv', auth)
