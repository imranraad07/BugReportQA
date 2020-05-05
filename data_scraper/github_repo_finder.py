import requests
import os
import time
from queries import *

headers = {"Authorization": "Bearer a4a37bc57f01dfef13d3c5f629dbc51800d554ca"}


def run_query(query_template, data_out, cursor_out, cursor_start='null'):
    cursor = cursor_start
    has_next_page = True
    failed_cnt = 0
    while has_next_page is True and failed_cnt < 20:
        print("")
        query = query_template.substitute(cursorStart=cursor)
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

        if request.status_code == 200:
            result = request.json()
            dump_data(result, data_out)
            cursor, has_next_page = get_cursor(result, cursor_out)
            cursor = "\""+cursor+"\""
            failed_cnt = 0
        elif request.status_code == 502:
            print("Query failed to run by returning code of 502. Try again in 30s...")
            time.sleep(30)
            failed_cnt += 1
            continue
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

    if failed_cnt > 0:
        raise Exception("Failed to process query starting after {0}".format(cursor))


def dump_data(result, data_out):
    if not os.path.exists(data_out):
        with open(data_out, 'w') as f:
            f.write('repo,url,commitNo,createdAt,lastCommit,bugReportsNo\n')

    with open(data_out, 'a') as f:
        for repo in result['data']['search']['edges']:
            url = repo['node']['url']
            name = repo['node']['name']
            created_at = repo['node']['createdAt']
            issue_no = int(repo['node']['issues']['totalCount'])
            if issue_no == 0:
                continue
            history = repo['node']['defaultBranchRef']['target']['history']
            commit_no = int(history['totalCount'])
            if commit_no == 0:
                continue
            last_commit_date = history['edges'][0]['node']['committedDate']
            f.write(
                ','.join([name, url, created_at, str(commit_no), created_at, last_commit_date, str(issue_no)]) + '\n')


def get_cursor(result, cursor_out):
    if not os.path.exists(cursor_out):
        with open(cursor_out, 'w') as f:
            f.write('cursor\n')

    has_next_page = result['data']['search']['pageInfo']['hasNextPage']
    cursor = result['data']['search']['pageInfo']['endCursor']

    with open(cursor_out, 'a') as f:
        f.write(cursor+"\n")

    return cursor, has_next_page


if __name__ == '__main__':
    run_query(repo_query, 'repos.csv', 'cursor.txt')
