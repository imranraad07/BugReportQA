import argparse
import collections
import csv
import sys

import numpy

from data_scraper.github_apis_v3 import get_issue_labels

headers = {'Authorization': 'token e0611cfcb582b98c9d94c3b53a380b5b88d98c2e',
           'Accept': 'application/vnd.github.mercy-preview+json'}


def read_github_repo_label(github_repo):
    try:
        url = "https://api.github.com/repos/{github_repo}"
        url = url.format(github_repo=github_repo)
        issue_data = get_issue_labels(url, headers)
        # print(issue_data)
        list_label = ' '.join([str(elem) for elem in issue_data['topics']])
        # print(list_label)
        return github_repo, list_label
    except Exception as ex:
        print("exception on issue, ", github_repo, str(ex))
    return None


def main(args):
    repos_with_labels = []
    labels = []
    for repo_csv_file in args.repos_csv:
        with open(repo_csv_file) as csvDataFile:
            csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
            print(next(csv_reader))
            for row in csv_reader:
                repo_labels = read_github_repo_label(row[1][19:])
                print(repo_labels)
                if repo_labels is not None:
                    for label in repo_labels[1].split():
                        labels.append(label)
                    repos_with_labels.append(repo_labels)

    allowed_labels = []
    print(len(labels))
    d = collections.Counter(numpy.array(labels))
    for item in d:
        if d[item] < 2:
            continue
        allowed_labels.append(item)
    print(len(allowed_labels))

    csv_file = open(args.output_csv, 'w')
    csv_writer = csv.writer(args.output_csv)

    for row in repos_with_labels:
        labels_list = row[1].split()
        repo_labels = []
        for label in labels_list:
            if label in allowed_labels:
                repo_labels.append(label)
        list_label = ' '.join([str(elem) for elem in repo_labels])
        csv_writer.writerow([row[0], list_label])
    csv_file.close()


def label_filterer(file_name):
    repos_with_labels = []
    labels = []
    with open(file_name) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        next(csvReader)
        for row in csvReader:
            repos_with_labels.append(row)
            for label in row[1].split():
                labels.append(label)

    allowed_labels = []
    print("total labels:", len(labels))
    d = collections.Counter(numpy.array(labels))
    print("unique labels:", len(d))
    for item in d:
        if d[item] < 2:
            continue
        allowed_labels.append(item)
    print("allowed labels:", len(allowed_labels))

    csv_file = open(file_name, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['repo', 'label'])

    for row in repos_with_labels:
        labels_list = row[1].split()
        repo_labels = []
        for label in labels_list:
            if label in allowed_labels:
                repo_labels.append(label)
        list_label = ' '.join([str(elem) for elem in repo_labels])
        csv_writer.writerow([row[0], list_label])
    csv_file.close()


if __name__ == '__main__':
    repos_csv_list = ['../data/repos/repos_final2008.csv',
                      '../data/repos/repos_final2009.csv',
                      '../data/repos/repos_final2010.csv',
                      '../data/repos/repos_final2011.csv',
                      '../data/repos/repos_final2012-part1.csv',
                      '../data/repos/repos_final2012-part2.csv',
                      '../data/repos/repos_final2013-part1.csv',
                      '../data/repos/repos_final2013-part2.csv',
                      '../data/repos/repos_final2014-part1.csv',
                      '../data/repos/repos_final2014-part2.csv'
                      ]
    output_file = '../data/bug_reports/github_repo_labels.csv'
    print(sys.argv)

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--repos_csv", type=list, default=repos_csv_list)
    argparser.add_argument("--output_csv", type=str, default=output_file)

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
    # label_filterer(args.output_csv)
