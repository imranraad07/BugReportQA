import time

import requests


def get_comments(url, headers):
    return execute_query(url, headers, 'Comments')


def get_an_issue(repo, issue_id, headers):
    url = "https://api.github.com/repos/{repo}/issues/{issue_id}"
    url = url.format(repo=repo, issue_id=issue_id)

    return execute_query(url, headers, 'Issues')


def execute_query(url, headers, query_type):
    response = requests.get(url, headers=headers)
    max_try = 20
    while response.status_code != 200:
        if response.status_code == 404:
            print("{2} - {0}. Bad response code 404 Not Found for {1}. Return None.".format(query_type, url,
                                                                                            time.ctime()))
            return None
        elif response.status_code == 410:
            print(
                "{2} - {0}. Bad response code: 410 Requested page is no longer available for {1}. Return None.".format(
                    query_type, url, time.ctime()))
            return None
        elif response.status_code == 301:
            print("{2} - {0}. Bad response code: 301 Requested page permanently moved for {1}. Return None.".format(
                query_type, url, time.ctime()))
            return None

        if max_try < 0:
            break
        max_try = max_try - 1
        if response.status_code == 401:
            print("{2} - {0}. Bad response code 401 Bad Credentials for {1}. Sleep for 3 minutes...".format(query_type,
                                                                                                            url,
                                                                                                            time.ctime()))
        else:
            print("{2} - {0}. Bad response code: {3} for {1}. Sleeping for 3 minutes....".format(query_type, url,
                                                                                                 time.ctime(),
                                                                                                 response.status_code))
        time.sleep(180)
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None
