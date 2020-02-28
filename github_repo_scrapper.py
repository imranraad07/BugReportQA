import json
import time

import requests


def get_repos(auth, file_name):
    step = 3000
    item_count = 0
    while step > 90:
        url = "https://api.github.com/search/repositories?q=stars:{star}+language:java"
        # url = "https://api.github.com/search/repositories?q=stars:>=2200+language:java"
        url = url.format(star=step)
        step = step - 1
        link = dict(next=url)
        # print(link, time.ctime())
        while 'next' in link:
            response = requests.get(link['next'], auth=auth)
            # print(link, response.status_code)

            while response.status_code != 200:
                # print("Bad response code:", response.status_code, "sleeping for 30 second....", time.ctime())
                time.sleep(30)
                response = requests.get(link['next'], auth=auth)

            # print(response.json())
            for item in response.json()['items']:
                print(item_count, item['full_name'], item['stargazers_count'], item['has_issues'])
                if item['has_issues'] is True:
                    item_count = item_count + 1
                    with open(file_name, "a") as myfile:
                        myfile.write(item['full_name'] + "\n")

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


if __name__ == '__main__':
    with open('credentials.json') as json_file:
        data = json.load(json_file)
    username = data['username']
    password = data['password']
    auth1 = (username, password)
    get_repos(auth1, 'github_repos.txt')