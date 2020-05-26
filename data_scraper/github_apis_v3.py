import time

import requests


def get_comments(url, headers):
    response = requests.get(url, headers=headers)
    max_try = 20
    while response.status_code != 200:
        if response.status_code == 404:
            print("Comments, Bad response code 404 Not Found, returning...", time.ctime())
            return None
        elif response.status_code == 410:
            print("Comments, Bad response code: 410 Requested page is no longer available. Returning...", time.ctime())
            return None
        elif response.status_code == 301:
            print("Comments, Bad response code: 301 Requested page permanently moved. Returning...", time.ctime())
            return None

        if max_try < 0:
            break
        max_try = max_try - 1
        if response.status_code == 401:
            print("Comments, Bad response code 401 Bad Credentials, sleeping for 3 minutes...", time.ctime())
        else:
            print("Comments, Bad response code:", response.status_code, "sleeping for 3 minutes....", time.ctime())
        time.sleep(180)
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_an_issue(repo, issue_id, headers):
    url = "https://api.github.com/repos/{repo}/issues/{issue_id}"
    url = url.format(repo=repo, issue_id=issue_id)

    response = requests.get(url, headers=headers)
    max_try = 20
    while response.status_code != 200:
        if response.status_code == 404:
            print("Issues, Bad response code: 404 Not Found. Returning...", time.ctime())
            return None
        elif response.status_code == 410:
            print("Issues, Bad response code: 410 Requested page is no longer available. Returning...", time.ctime())
            return None
        elif response.status_code == 301:
            print("Issues, Bad response code: 301 Requested page permanently moved. Returning...", time.ctime())
            return None

        if max_try < 0:
            break
        max_try = max_try - 1
        if response.status_code == 401:
            print("Issues, Bad response code 401 Bad Credentials, sleeping for 3 minutes...", time.ctime())
        else:
            print("Issues, Bad response code:", response.status_code, "sleeping for 3 minutes....", time.ctime())
        time.sleep(180)
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        _json = response.json()
        return _json
    else:
        return None
