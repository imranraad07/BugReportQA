import requests
import os
import time
from queries import *
from datetime import datetime, timedelta

date = datetime.strptime('Apr 1 2008', '%b %d %Y')
headers = {"Authorization": "Bearer a4a37bc57f01dfef13d3c5f629dbc51800d554ca"}


def run_query(query_template, data_out, params_out, created_at, interval=14):
    cursor = 'null'
    has_next_page = True
    failed_cnt = 0
    start_date = created_at.strftime("%Y-%m-%d")
    end_date = (created_at + timedelta(days=interval)).strftime("%Y-%m-%d")

    while has_next_page is True and failed_cnt < 20:
        query = query_template.substitute(cursorStart=cursor, start=start_date, end=end_date)
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

        if request.status_code == 200:
            failed_cnt = 0
            result = request.json()

            if limit_exceeded(result, interval) is True:
                # decrease time interval and try again
                print('Starting {0} the interval of {1} days is too big. Changed to {2} days.'.format(start_date,
                                                                                                      interval,
                                                                                                      interval - 1))
                interval -= 1
                end_date = (created_at + timedelta(days=interval)).strftime("%Y-%m-%d")
                continue

            dump_data(result, data_out)
            save_query_params(params_out, start_date, end_date, cursor)
            cursor, has_next_page = get_cursor(result)
            cursor = "\"" + cursor + "\""

        elif request.status_code == 502:
            print("Query failed to run by returning code of 502 for cursor {0}. Try again in 30s...".format(cursor))
            time.sleep(30)
            failed_cnt += 1
            continue
        else:
            raise Exception(
                "Query failed to run by returning code of {0}. Query params: created:{0}..{1} cursor:{2}.".format(
                    request.status_code, start_date, end_date, cursor))

    if failed_cnt > 0:
        raise Exception(
            "Failed to process query with params: created:{0}..{1} cursor:{2}.".format(start_date, end_date, cursor))

    repository_count = result['data']['search']['repositoryCount']
    return repository_count, interval


def dump_data(result, data_out):
    if not os.path.exists(data_out):
        with open(data_out, 'w') as f:
            f.write('repo,url,commitNo,createdAt,lastCommit,bugReportsNo\n')

    cnt = 0
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
            cnt += 1

    print('Saved data from {0} repositories.'.format(cnt))


def get_cursor(result):
    has_next_page = result['data']['search']['pageInfo']['hasNextPage']
    cursor = result['data']['search']['pageInfo']['endCursor']

    return cursor, has_next_page


def save_query_params(cursor_out, start_date, end_date, cursor):
    if not os.path.exists(cursor_out):
        with open(cursor_out, 'w') as f:
            f.write('start_date,end_date,cursor\n')

    with open(cursor_out, 'a') as f:
        f.write('{0},{1},{2}\n'.format(start_date, end_date, cursor))


def limit_exceeded(result, interval):
    if result['data']['search']['repositoryCount'] > 1000 or interval > 1:
        return False
    return True


if __name__ == '__main__':
    start_date = datetime.strptime('Apr 1 2008', '%b %d %Y')
    repo_cnt = 0
    interval = 14
    while repo_cnt < 30000000:
        print('Query for repos created at: {0} - {1}'.format(start_date, start_date + timedelta(days=interval)))
        cnt, interval = run_query(repo_query, 'repos.csv', 'params.csv', start_date, interval)

        start_date = start_date + timedelta(days=interval)
        repo_cnt += cnt
