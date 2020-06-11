import argparse
import csv
import sys

# import nltk
# nltk.download('punkt')
from data_scraper.github_apis_v3 import get_issue_labels

headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e',
           'Accept': 'application/vnd.github.mercy-preview+json'}


def read_github_issues(github_repo, csv_writer):
    try:
        url = "https://api.github.com/repos/{github_repo}"
        url = url.format(github_repo=github_repo)
        issue_data = get_issue_labels(url, headers)
        # print(issue_data)
        list_label = ' '.join([str(elem) for elem in issue_data['topics']])
        # print(list_label)
        write_row = [github_repo, list_label]
        csv_writer.writerow(write_row)
    except Exception as ex:
        print("exception on issue, ", github_repo, str(ex))


def parse_repos(input_file, result_file):
    if 'write' in args.type:
        csv_file = open(result_file, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['repo', 'label'])
    else:
        csv_file = open(result_file, 'a')
        csv_writer = csv.writer(csv_file)

    with open(input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            read_github_issues(row[1][19:], csv_writer)
    csv_file.close()


def main(args):
    csv.field_size_limit(sys.maxsize)
    parse_repos(args.repo_csv, args.output_csv)


if __name__ == '__main__':
    print(sys.argv)

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--type", type=str, default='write') # write/append

    argparser.add_argument("--repo_csv", type=str,
                           default='../data/repos/repos_final2008.csv')
    argparser.add_argument("--output_csv", type=str,
                           default='../data/bug_reports/github_br_labels.csv')

    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
