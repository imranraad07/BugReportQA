import logging
import time
from datetime import datetime, timedelta
from string import Template

import requests
from nltk import sent_tokenize

from data_generation.calculator import Calculator
from data_scraper.github_repo_finder import get_contributors
from utils import question_identifier

logger = logging.getLogger('follow-up-questions')
headers = {"Authorization": "Bearer a4a37bc57f01dfef13d3c5f629dbc51800d554ca"}

repos = [
    'golang/go', 'kubernetes/kubernetes', 'rust-lang/rust', 'ansible/ansible', 'microsoft/TypeScript',
    'elastic/elasticsearch', 'godotengine/godot', 'saltstack/salt', 'angular/angular', 'moby/moby']

query_template = Template("""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  repository(owner: "$owner", name: "$name") {
   issues(first: $n, filterBy: {states:CLOSED, labels: ["bug", "Bug", ">bug", "crash", "type: bug/fix","kind/bug", "WaitingForInfo", "UX", "NeedsInvestigation", "NeedsFix", "kind/bug", "C-bug", "I-ICE"]}){
    totalCount
    nodes{
      ... on Issue{
        number
        title
        body
        createdAt
        author {
          login
        }
        comments(first:20){
          nodes{
            ... on IssueComment{
              author{
                login
              }
              createdAt
              body
            }
          }
        }
      }
    }
  } 
  }
}
""")


def extract_data(issue):
    comment_count = 0
    after_question = 0
    has_follow_up_question = False
    has_follow_up_answer = False
    is_OB_EB_S2R = False
    follow_up_question = ''
    follow_up_question_reply = ''

    if len(issue['comments']['nodes']) < 2 or issue['body'] is None or issue['author'] is None:
        return None

    comments = issue['comments']['nodes']
    for comment in comments:
        # comment within 60 days of issue creation
        d1 = datetime.strptime(comment['createdAt'], "%Y-%m-%dT%H:%M:%SZ")
        d2 = datetime.strptime(issue['createdAt'], "%Y-%m-%dT%H:%M:%SZ")
        if d1 - d2 > timedelta(days=60):
            continue

        if not has_follow_up_question and comment_count < 3:
            if comment['author'] is None:
                continue
            comment_count = comment_count + 1
            # if comment author and issue author are same, then discard the comment
            if comment['author']['login'] == issue['author']['login']:
                continue
            follow_up_question = comment['body']
            for sentence in sent_tokenize(comment['body']):
                sentence = sentence.strip()
                if sentence.startswith(">") is True:
                    continue
                elif question_identifier(sentence) is True:
                    # if sentence starts with @someone, check if this @someone is original issue author or not
                    if sentence.startswith("@"):
                        mentioned_login = sentence.split()[0]
                        github_login = "@{0}".format(issue['author']['login'])
                        if mentioned_login != github_login:
                            break
                    has_follow_up_question = True
                    idx = comment['body'].find(sentence)
                    follow_up_question = comment['body'][idx:]
                    after_question = 0
                    break
        elif has_follow_up_question and after_question < 3:
            after_question = after_question + 1
            if issue['author']['login'] == comment['author']['login']:
                has_follow_up_answer = True
                follow_up_question_reply = comment['body']
                break

    if has_follow_up_question and has_follow_up_answer:
        # filter by word count and #characters
        comment_array = follow_up_question.split()
        if len(comment_array) > 30 or len(follow_up_question) > 300:
            return False, False, False

        calc = Calculator(threshold=0)
        if calc.utility(issue['number'], follow_up_question_reply, '') > 0:
            is_OB_EB_S2R = True

    return has_follow_up_question, has_follow_up_answer, is_OB_EB_S2R


def get_issues(owner, repo, n):
    failed_cnt = 0
    while failed_cnt < 20:
        query = query_template.substitute(owner=owner, name=repo, n=n)
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

        if request.status_code == 200:
            return request.json()['data']['repository']['issues']['nodes']
        elif request.status_code == 502:
            logger.info("Query failed to run by returning code of 502. Try again in 30s...")
            time.sleep(30)
            failed_cnt += 1
            continue
        elif request.status_code == 403:
            logger.info('Abusive behaviour mechanism was triggered. Wait 3 min.')
            time.sleep(180)
            failed_cnt += 1
            continue
        else:
            raise Exception(
                "Query failed to run by returning code of {0}.".format(request.status_code))

    if failed_cnt == 20:
        logger.info('Cannot process query {0}'.format(query))


# R = the set of top 10 (listed below) repositories:
# for each repo's r in R:
#     B = set of 50 most recently reported issues
#     for each issue b in B:
#         y = determine if b contains a follow-up question
#         if y:
#             x = determine if the follow-up question is answered on unanswered
#             if x:
#                  z = does the answer to the follow-up question contain OB, EB or S2R
# return x, y, z for each r


if __name__ == '__main__':
    n = 100
    results = dict()

    for repo in repos:
        logger.info('Processing {0}'.format(repo))
        results[repo] = (list(), list(), list())
        owner, name = repo.split('/')
        issues = get_issues(owner, name, n)
        #contributors = get_contributors('///'+repo)
        for issue in issues:

            #filter out issues created by devs
            if issue['author'] is None: #or issue['author']['login'] in contributors:
                continue

            data = extract_data(issue)
            if data is None:
                continue
            results[repo][0].append(data[0])
            results[repo][1].append(data[1])
            results[repo][2].append(data[2])

        # plot(results['repo'])
        # print('{0}'.format(results[repo]))
        print(
            'Repo {0} has {1}/{2} follow-up questions, {3}/{1} has been answered, {4}/{3} answers has OB/EB/S2R'.format(
                repo, sum(results[repo][0]), len(results[repo][0]), sum(results[repo][1]), sum(results[repo][2])))
