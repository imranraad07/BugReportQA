import csv
import sys
from datetime import datetime, timedelta

import argparse
import difflib

from nltk import sent_tokenize

from github_apis_v3 import *
from github_text_filter import *
from queries import *
from utils import question_identifier

# import nltk
# nltk.download('punkt')

headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e'}
headers1 = {'Authorization': 'token 915e33be5b0f3aca5f8a9b5d621e4bfd27177145'}
headers2 = {"Authorization": 'token e551fea0cc4c810adf0b5bb3e04faa9eee6327c9'}
headers3 = {'Authorization': 'token caaa28ea2380831278a1cdc3378fc0d01ccb0062'}
headers4 = {'Authorization': 'token ba06210a21cbae85560b1498e9e98bc2d736979a'}

counter_header = 0
run_now = 0
bug_report_counter = 0


def getHeader():
    global counter_header
    counter_header = counter_header + 1
    if counter_header % 5 == 0:
        return headers1
    elif counter_header % 5 == 1:
        return headers2
    elif counter_header % 5 == 2:
        return headers3
    elif counter_header % 5 == 3:
        return headers4
    return headers


def read_github_issues(github_repo, bug_ids, csv_writer):
    global bug_report_counter
    for issue_id in bug_ids:
        print("issue_id", issue_id, "BR count", bug_report_counter)
        try:
            issue_data = get_an_issue(github_repo, issue_id, getHeader())
            # print(issue_data)

            if issue_data is None:
                continue

            # github v3 api considers pull requests as issues. so filter them
            if 'pull_request' in issue_data:
                continue
            if 'comments' not in issue_data:
                print("comments is not in issue data")
                continue
            # check if comment count is at least two
            if issue_data['comments'] < 2:
                continue

            comment_count = 0
            after_question = 0
            is_follow_up_question = False
            is_follow_up_question_answer = False
            follow_up_question = ''
            follow_up_question_reply = ''
            if 'comments_url' not in issue_data:
                print("comments_url is not in issue data")
                continue
            comments = get_comments(issue_data['comments_url'], getHeader())
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
                                mentioned_login = sentence.split()[0]
                                github_login = "@{0}".format(issue_data['user']['login'])
                                if mentioned_login != github_login:
                                    break
                            is_follow_up_question = True
                            idx = comment['body'].find(sentence)
                            follow_up_question = comment['body'][idx:]
                            after_question = 0
                            break
                elif is_follow_up_question and after_question < 3:
                    after_question = after_question + 1
                    if issue_data['user']['login'] == comment['user']['login']:
                        follow_up_question_reply = comment['body']
                        is_follow_up_question_answer = True
                        break

            if is_follow_up_question and is_follow_up_question_answer:
                # filter by word count and #characters
                comment_array = follow_up_question.split()
                if len(comment_array) > 30 or len(follow_up_question) > 300:
                    continue

                original_post = filter_nontext(issue_data['title'])
                if issue_data['body'] is not None:
                    original_post = original_post + "\n\n" + filter_nontext(issue_data['body'])
                follow_up_question = filter_nontext(follow_up_question)
                follow_up_question_reply = filter_nontext(follow_up_question_reply)
                postid = issue_data['html_url'][19:]
                postid = postid.replace("/", "_")
                write_row = [github_repo, issue_data['html_url'], postid, original_post, follow_up_question,
                             follow_up_question_reply]
                csv_writer.writerow(write_row)
                print(write_row)
                bug_report_counter = bug_report_counter + 1
        except Exception as ex:
            print("exception on issue, ", issue_id, str(ex))


def parse_repos(input_file, result_file):
    csv_file = open(result_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])

    with open(input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            read_github_issues(row[1][19:], row[9:], csv_writer)
    csv_file.close()


#############################################################################################################
#############################################################################################################

def get_edits(repo_url, issue_no):
    tokens = repo_url.split('/')
    owner = '\"' + tokens[3] + '\"'
    name = '\"' + tokens[4] + '\"'

    query = edit_query.substitute(owner=owner, name=name, number=issue_no)
    failed_cnt = 0
    while failed_cnt < 20:
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=getHeader())

        if request.status_code == 200:
            result = request.json()
            if is_error(result) is True:
                failed_cnt += 1
                continue
            edits = [(x['node']['diff'], x['node']['createdAt']) for x in
                     result['data']['repository']['issue']['userContentEdits']['edges']]
            # if len(edits) > 1:
            #     print(len(edits))
            return edits
        elif request.status_code == 502:
            print("Query failed to run by returning code of 502 for repo {0} - issue {1}. Try again in 30s...".format(
                repo_url, issue_no))
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
                "Query failed to run by returning code of {0}. Query params: url:{1}, issue:{2}.".format(
                    request.status_code, repo_url, issue_no))


def is_error(result):
    slept = False
    if 'errors' in result:
        for error in result['errors']:
            if 'type' in error:
                if error['type'] == 'RATE_LIMITED':
                    print('API rate limit exceeded. Wait 10 min...')
                    time.sleep(600)
                    slept = True
            elif 'message' in error:
                print(error['message'])

        # if slept is False:
        #     print('Dont know these errors. Lets sleep 3 min just in case!')
        #     time.sleep(180)
        return True

    return False


def get_follow_up_question(issue):
    if 'comments' not in issue:
        print("comments is not in issue data")
        return None

    comment_count = 0
    is_follow_up_question = False
    follow_up_question = ''
    follow_up_question_time = None
    if 'comments_url' not in issue:
        print("comments_url is not in issue data")
        return None
    comments = get_comments(issue['comments_url'], getHeader())
    if comments is None:
        return None

    # reading the comments
    for comment in comments:
        if 'user' not in comment:
            print("user is not in comment data")
            continue
        if 'user' not in issue:
            print("user is not in issue data")
            continue

        # comment within 60 days of issue creation
        d1 = datetime.strptime(comment['created_at'], "%Y-%m-%dT%H:%M:%SZ")
        d2 = datetime.strptime(issue['created_at'], "%Y-%m-%dT%H:%M:%SZ")
        if d1 - d2 > timedelta(days=60):
            # print(d1-d2)
            continue

        if not is_follow_up_question and comment_count < 3:
            comment_count = comment_count + 1
            # if comment author and issue author are same, then discard the comment
            if comment['user']['id'] == issue['user']['id']:
                continue
            follow_up_question = comment['body']
            follow_up_question_time = comment['created_at']
            for sentence in sent_tokenize(comment['body']):
                sentence = sentence.strip()
                if sentence.startswith(">"):
                    continue
                elif question_identifier(sentence):
                    # if sentence starts with @someone, check if this @someone is original issue author or not
                    if sentence.startswith("@"):
                        is_mentioned = sentence.split()[0]
                        github_login = "@{0}".format(issue['user']['login'])
                        if is_mentioned != github_login:
                            break
                    is_follow_up_question = True
                    idx = comment['body'].find(sentence)
                    follow_up_question = comment['body'][idx:]
                    break

    if is_follow_up_question:
        # just filtering by character count
        comment_array = follow_up_question.split()
        if len(comment_array) > 30 or len(follow_up_question) > 300:
            return None
        follow_up_question = filter_nontext(follow_up_question)
        return follow_up_question, follow_up_question_time
    return None


def show_diff(text, n_text):
    seqm = difflib.SequenceMatcher(None, text, n_text)
    output = []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'insert':
            output.append(seqm.b[b0:b1])
        else:
            continue
    return ''.join(output)


# steps:
# 1. get edit list
# 2. get question comment
# 3. check if any edit exists after question comment time. if exists, this might be the answer
# 4. check difference between this edit and original post(or edited post right before this edit)
def get_edit_by_issue(repo_url, issue_id, csv_writer):
    try:
        # print(repo_url, issue_id)

        # step 1: get edits
        response = get_edits(repo_url, issue_id)
        if response is None:
            return

        if len(response) > 0:
            # step 2: get follow up question
            issue = get_an_issue(repo_url[19:], issue_id, getHeader())
            if issue is None:
                return

            follow_up_question = get_follow_up_question(issue)
            if follow_up_question is None:
                return

            # response is time sorted
            # adding the current post to response edit list(ie this is the latest post)
            response.insert(0, (issue['body'], issue['updated_at']))

            idx = len(response) - 2
            for i in range(0, len(response) - 1):
                edit = response[idx]
                prev_edit = response[idx + 1]

                if edit[0] is None or prev_edit[0] is None:
                    continue

                d1 = datetime.strptime(follow_up_question[1], "%Y-%m-%dT%H:%M:%SZ")
                d2 = datetime.strptime(edit[1], "%Y-%m-%dT%H:%M:%SZ")
                print(d1, d2, d1 < d2)

                # step 4: check time exists (the first one right after the follow_up_question)
                if d1 < d2:
                    original_text = prev_edit[0]
                    modified_text = edit[0]

                    # step 5: check difference
                    diff = show_diff(original_text, modified_text)
                    if not diff:
                        continue
                    diff = filter_nontext(diff.strip())
                    if len(diff.split()) < 4:
                        continue

                    print("----------------compare string------------------")
                    print("diff: ", diff)

                    postid = issue['html_url'][19:]
                    postid = postid.replace("/", "_")
                    original_post = filter_nontext(issue['title'])
                    if issue['body'] is not None:
                        original_post = original_post + "\n\n" + filter_nontext(issue['body'])
                    write_row = [repo_url[19:], issue['html_url'], postid, original_post.strip(),
                                 follow_up_question[0].strip(), diff]
                    csv_writer.writerow(write_row)
                    print(write_row)
                    global bug_report_counter
                    bug_report_counter = bug_report_counter + 1
                    print("Edit Bug Reports:", bug_report_counter)
                    break
                idx = idx - 1
    except Exception as e:
        print("Exception occurred", str(e))


def parse_edits(input_file, result_file):
    csv_file = open(result_file, 'a')
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])

    global run_now
    global bug_report_counter
    with open(input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            for issue_id in row[9:]:
                if "elastic/elasticsearch" in row[1]:
                    if int(issue_id) == 7365:
                        run_now = 1
                        print("run now is 1")
                    print("true", issue_id, run_now)
                if run_now == 0:
                    continue
                print(row[1], issue_id, bug_report_counter)
                get_edit_by_issue(row[1], issue_id, csv_writer)
    csv_file.close()


def main(args):
    type = args.type
    if type == "parse":
        parse_repos(args.repo_csv, args.output_csv)
    elif type == "edit":
        parse_edits(args.repo_csv, args.output_csv)


# arguments (type[edit/parse], input_repo_csv, output_folder, output_csv)
if __name__ == '__main__':
    print(sys.argv)

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--type", type=str, default='edit')
    argparser.add_argument("--repo_csv", type=str,
                           default='/home/imranm3/projects/BugReportQA/data/repos/repos_final2010.csv')
    argparser.add_argument("--output_csv", type=str,
                           default='/home/imranm3/projects/BugReportQA/data/bug_reports/github_data_2010_edit.csv')

    args = argparser.parse_args()
    print(args)
    print("")
    main(args)