import argparse
import csv
import sys

from data_scraper.github_apis_v3 import get_an_issue

headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e',
           'Accept': 'application/vnd.github.mercy-preview+json'}


def read_github_issue(issue_id):
    try:
        url = "https://api.github.com/repos/{issue_id}"
        url = url.format(issue_id=issue_id)
        issue = get_an_issue(url, headers)
        title = issue['title']
        return title
    except Exception as ex:
        print("exception on issue, ", issue_id, str(ex))
    return None


def main(args):
    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['issue_id', 'title'])
    with open(args.input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            issue_title = read_github_issue(row[1][19:])
            print(issue_title)
            csv_writer.writerow([row[2], issue_title])
    csv_file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str, default='../data/datasets/github/dataset.csv')
    argparser.add_argument("--output_file", type=str, default='../data/bug_reports/github_issue_titles.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
