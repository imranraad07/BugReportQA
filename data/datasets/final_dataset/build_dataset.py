import argparse
import csv
import sys


def main(args):
    ids = []
    issue_reports_all = {}
    with open(args.input_dataset_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            ids.append(row[2])
            issue_reports_all[row[2]] = row
    print(len(ids))
    bug_reports = {}
    with open(args.input_tag_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        for row in csv_reader:
            if row[0] in ids:
                if 'bug' in row[1] or 'defect' in row[1] or 'crash' in row[1] or 'fix' in row[1]:
                    if issue_reports_all[row[0]][0] in bug_reports:
                        bug_reports[issue_reports_all[row[0]][0]].append(issue_reports_all[row[0]])
                    else:
                        bug_reports[issue_reports_all[row[0]][0]] = []
                        bug_reports[issue_reports_all[row[0]][0]].append(issue_reports_all[row[0]])
                elif not row[1]:
                    if issue_reports_all[row[0]][0] in bug_reports:
                        bug_reports[issue_reports_all[row[0]][0]].append(issue_reports_all[row[0]])
                    else:
                        bug_reports[issue_reports_all[row[0]][0]] = []
                        bug_reports[issue_reports_all[row[0]][0]].append(issue_reports_all[row[0]])
    print(len(bug_reports))

    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['repo', 'issue_link', 'issue_id', 'post', 'question', 'answer'])
    final_br_list = []
    with open(args.input_repos_nonc) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        count = 0
        idx = 0
        for row in csv_reader:
            if row[0] in bug_reports:
                this_repo = 0
                for bug_report in bug_reports[row[0]]:
                    final_br_list.append(bug_report)
                    csv_writer.writerow(bug_report)
                    count = count + 1
                    this_repo = this_repo + 1
                    if count >= 25000:
                        break
                idx = idx + 1
                print(idx, row[0], this_repo)
                if count >= 25000:
                    break
    csv_file.close()
    print(len(final_br_list))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_dataset_file", type=str, default='dataset_40k.csv')
    argparser.add_argument("--input_tag_file", type=str, default='../bug_reports/github_issue_labels.csv')
    argparser.add_argument("--input_repos_nonc", type=str, default='../repos/repos_nonc.csv')
    argparser.add_argument("--output_file", type=str, default='dataset.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
