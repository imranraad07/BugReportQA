import requests
import json
import nltk

nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
import csv


def get_comments(url, auth):
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.json()


def get_issues(repo, auth):
    url = "https://api.github.com/repos/{repo}/issues?state=closed&sort=comments-desc"
    url = url.format(repo=repo)
    return _getter(url, auth)


def _getter(url, auth):
    link = dict(next=url)
    print(link)
    while 'next' in link:
        response = requests.get(link['next'], auth=auth)
        # And.. if we didn't get good results, just bail.
        if response.status_code != 200:
            print ("Bad response code: ", response.status_code)
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


if __name__ == '__main__':
    with open('credentials.json') as json_file:
        data = json.load(json_file)
    username = data['username']
    password = data['password']
    auth = (username, password)

    github_repos = [
        "elastic/elasticsearch",
        "spring-projects/spring-boot",
        "iluwatar/java-design-patterns",
        "ReactiveX/RxJava",
        "mockito/mockito",
        "google/guava",
        "square/okhttp"
        # "duckduckgo/Android", "PhilippC/keepass2android", "zxing/zxing", "AntennaPod/AntennaPod",
        # "brave/browser-android-tabs", "Telegram-FOSS-Team/Telegram-FOSS"
    ]

    totalIssues = 0
    for repo in github_repos:
        commentAddedCSVCount = 0
        issueCount = 0
        for issue in get_issues(repo, auth):
            issueCount = issueCount + 1
            totalIssues = totalIssues + 1
            # print(totalIssues)
            # consider at most 1000 issues
            # if issueCount > 1000:
            #     break

            # label bug check
            isLabelBug = False
            for label in issue['labels']:
                labelText = label['name']
                if "bug" in labelText:
                    isLabelBug = True
                    break

            if "bug" in issue['title']:
                isLabelBug = True

            if "bug" in issue['body']:
                isLabelBug = True

            if not isLabelBug:
                continue

            # print(commentAddedCSVCount, " ", issue['comments'], " ", issue['labels'], " ", issue['comments_url'])
            commentCount = 0
            followUpQuestion = False
            comments = get_comments(issue['comments_url'], auth)
            for comment in comments:
                commentCount = commentCount + 1
                if commentCount > 3:
                    break
                # if comment author and issue author are same, then discard the comment
                if comment['user']['id'] == issue['user']['id']:
                    continue
                for sentence in sent_tokenize(comment['body']):
                    if check(sentence):
                        commentAddedCSVCount = commentAddedCSVCount + 1
                        sw = csv.writer(open('results/data_github.csv', 'a'))
                        sw.writerow([
                            '{0}'.format(repo),
                            '{0}'.format(comment['html_url']),
                            '{0}'.format(comment['body'])
                        ])
                        followUpQuestion = True
                        break
                if followUpQuestion:
                    break
            # # at most 10 comments from each repo
            # if commentAddedCSVCount == 10:
            #     break
        print(totalIssues)