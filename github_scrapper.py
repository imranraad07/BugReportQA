import requests
import json
from nltk import word_tokenize, sent_tokenize
import csv


def get_comments(auth, url):
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.json()


def get_issues(auth):
    # url = "https://api.github.com/repos/flutter/flutter/issues?state=closed&sort=comments-desc"
    # url = "https://api.github.com/repos/google/material-design-icons/issues?state=closed&sort=comments-desc"
    # url = "https://api.github.com/repos/android/architecture-samples/issues?state=closed&sort=comments-desc"
    # url = "https://api.github.com/repos/square/okhttp/issues?state=closed&sort=comments-desc"
    # url = "https://api.github.com/repos/PhilJay/MPAndroidChart/issues?state=closed&sort=comments-desc"
    return _getter(url, auth)


def _getter(url, auth):
    link = dict(next=url)
    print(link)
    while 'next' in link:
        response = requests.get(link['next'], auth=auth)
        # And.. if we didn't get good results, just bail.
        # if response.status_code != 200:
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
    count = 0
    for issue in get_issues(auth):
        comments = get_comments(auth, issue['comments_url'])
        print(count, " ", issue['comments'], " ", len(comments), " ", issue['comments_url'])
        count = count + 1
        commentCount = 0
        followUpQuestion = False
        for comment in comments:
            commentCount = commentCount + 1
            if commentCount >= 3:
                break
            for sentence in sent_tokenize(comment['body']):
                if check(sentence):
                    count = count + 1
                    sw = csv.writer(open('results/data_github.csv', 'a'))
                    sw.writerow([
                        '{0}'.format(comment['html_url']),
                        '{0}'.format(sentence)
                    ])
                    followUpQuestion = True
                    break
            if followUpQuestion:
                break

        if count >= 100:
            break
