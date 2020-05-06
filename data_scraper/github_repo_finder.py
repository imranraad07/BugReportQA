import requests
import os
import sys
import time
import pandas as pd
from queries import *
from datetime import datetime, timedelta
import click

headers = {"Authorization": "Bearer a4a37bc57f01dfef13d3c5f629dbc51800d554ca"}


@click.command()
@click.option('--collect-repos', is_flag=True)
@click.option('--start-date', type=str, default='Apr 1 208')
@click.option('--interval', type=int, default=30)
@click.option('--repos-file', type=str, default='repos.csv')
@click.option('--params-file', type=str, default='params.csv')
@click.option('--process-repos', is_flag=True)
@click.option('--process-repos-out', type=str, default='repos_processed.csv')
def run(*args, **kwargs):
    repos_fpath = kwargs['repos_file']
    params_fpath = kwargs['params_file']

    if kwargs['collect_repos'] is True:
        start_date = kwargs['collect_repos']
        interval = kwargs['interval']
        collect_repos(repos_fpath, params_fpath, start_date, interval)
    if kwargs['process_repos']:
        out_fpath = kwargs['process_repos_out']
        process_repos(repos_fpath, out_fpath)
        bugs_by_non_contributors(out_fpath)


def collect_repos(repos_fpath, params_fpath, start_date, days_interval):
    if not os.path.exists(repos_fpath):
        start = datetime.strptime(start_date, '%b %d %Y')
        interval = days_interval
        while start < datetime.now() - timedelta(days=10):
            print('Query for repos created at: {0} - {1}'.format(start, start + timedelta(days=interval)))
            cnt, interval = run_query(repo_query, repos_fpath, params_fpath, start, interval)
            start = start + timedelta(days=interval)


def run_query(query_template, data_out, params_out, created_at, interval=14, cursor='null'):
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
        elif request.status_code == 403:
            print('Abusive behaviour mechanism was triggered. Wait 3 min.')
            time.sleep(180)
            failed_cnt += 1
            continue
        else:
            raise Exception(
                "Query failed to run by returning code of {0}. Query params: created:{1}..{2} cursor:{3}.".format(
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
                ','.join([name, url, str(commit_no), created_at, last_commit_date, str(issue_no)]) + '\n')
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
    if result['data']['search']['repositoryCount'] > 1000 and interval > 1:
        return True
    return False


def process_repos(repos_file, repos_out_file):
    df = pd.read_csv(repos_file)
    df['start_epoch'] = df['createdAt'].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds())
    df['end_epoch'] = df['lastCommit'].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1, 1)).total_seconds())
    df['days'] = (df['end_epoch'] - df['start_epoch']) / float(86400)

    df = df.drop(columns=['start_epoch', 'end_epoch'])
    df['bugsPerDayAll'] = df['bugReportsNo'] / df['days']
    df = df.sort_values(by=['bugsPerDayAll'], ascending=False).reset_index(drop=True)
    df.to_csv(repos_out_file, index=False)
    print('Done. Processed file saved to {0}'.format(repos_out_file))


def bugs_by_non_contributors(repos_file):
    out_file = list()
    try:
        with open(repos_file, 'r') as f:
            # skip header
            f.readline()
            for line in f.readlines():
                repo, url, commitNo, createdAt, lastCommit, bugReportsNo, days, bugsPerDayAll = line.split(',')
                print('Process repo {0}'.format(url))

                contributors = get_contributors(url)
                author2br = get_br_creators_emails(url)
                author2br_filtered = cross_reference(contributors, author2br)
                cnt = sum([len(author2br_filtered[key]) for key in author2br_filtered])
                bids = list()
                for key in author2br_filtered:
                    bids.extend(author2br_filtered[key])
                out_file.append(line.strip() + ',' + str(cnt) + ',' + str(cnt / float(days)) + ',' + ','.join(
                    [str(x) for x in bids]))
    except Exception as e:
       print('Exception occurred. Too bad. Save what we collected so far. Exception: {0}'.format(e))

    with open(repos_file, 'w') as f:
        f.write(
            'repo,url,commitNo,createdAt,lastCommit,bugReportsNo,days,bugsPerDayAll,bugReportsByNonC,bugsPerDayNonC,BIDs\n')
        for line in out_file:
            f.write(line + '\n')

    print('Finished processing file. Save to {0}'.format(repos_file))


def get_contributors(url):
    tokens = url.split('/')
    owner = tokens[3]
    name = tokens[4]
    query = 'https://api.github.com/repos/{0}/{1}/contributors'.format(owner, name)
    failed_cnt = 0
    while failed_cnt < 20:
        request = requests.get(query)
        if request.status_code == 200:
            result = request.json()
            contributors = set()
            for user in result:
                contributors.add(user['login'])
            print('Contributors collected')
            return contributors
        elif failed_cnt < 20:
            print('Cannot get contributors for repo {0}. Try again in 10s...'.format(url))
            failed_cnt += 1
        else:
            raise Exception(
                'Failed to process query: {0}\n Status code {1}. Request {2}.'.format(query, request.status_code,
                                                                                      request))


def get_br_creators_emails(url):
    print('Get authors of bug reports for repo {0}'.format(url))
    tokens = url.split('/')
    owner = '\"' + tokens[3] + '\"'
    name = '\"' + tokens[4] + '\"'

    cursor = 'null'
    author2br = dict()
    has_next_page = True
    failed_cnt = 0
    while has_next_page is True and failed_cnt < 20:
        query = issues_query.substitute(owner=owner, name=name, cursor=cursor)
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

        if request.status_code == 200:
            failed_cnt = 0
            result = request.json()
            author2br = process_issues(result, author2br)

            cursor = '\"' + result['data']['repository']['issues']['pageInfo']['endCursor'] + '\"'
            has_next_page = result['data']['repository']['issues']['pageInfo']['hasNextPage']
        elif request.status_code == 502:
            print("Query to repo {0} failed to run by returning code of 502 for cursor {0}. Try again in 30s...".format(
                url, cursor))
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
                "Query failed to run by returning code of {0}. Query params: owner: {1}, name {2}, cursor {3}".format(
                    request.status_code, owner, name, cursor))
    print('Collected bug reports authors')
    return author2br


def process_issues(result, author2br):
    for node in result['data']['repository']['issues']['edges']:
        issue = node['node']
        no = issue['number']
        if issue['author'] is not None:
            author = issue['author']['login']
        else:
            author = 'ghost'
        if author not in author2br:
            author2br[author] = list()
        author2br[author].append(no)

    return author2br


def cross_reference(contributors, author2br):
    for contr in contributors:
        if contr in author2br:
            del author2br[contr]
    return author2br


if __name__ == '__main__':
    run(sys.argv[1:])
