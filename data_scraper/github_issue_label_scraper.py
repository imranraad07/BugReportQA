import argparse
import csv
import sys

from data_scraper.github_apis_v3 import get_issue_labels

headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e',
           'Accept': 'application/vnd.github.mercy-preview+json'}


def read_github_issue_label(issue_id):
    try:
        url = "https://api.github.com/repos/{issue_id}/labels"
        url = url.format(issue_id=issue_id)
        issue_labels = get_issue_labels(url, headers)
        labels = ''
        for label in issue_labels:
            labels = labels + " " + label['name']
        return labels
    except Exception as ex:
        print("exception on issue, ", issue_id, str(ex))
    return None


def main(args):
    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['issue_id', 'label'])
    with open(args.input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            repo_labels = read_github_issue_label(row[1][19:])
            print(repo_labels)
            csv_writer.writerow(row[2], repo_labels)
    csv_file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str, default='../data/bug_reports/github_data_2008.csv')
    argparser.add_argument("--output_file", type=str, default='../data/bug_reports/github_issue_labels.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
