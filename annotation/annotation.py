import argparse
import csv
import sys

# golang/go, kubernetes/kubernetes, rust-lang/rust, ansible/ansible, microsoft/TypeScript, elastic/elasticsearch,
# godotengine/godot, saltstack/salt, angular/angular, moby/moby
import random


def main(args):
    issues = {}
    with open(args.input_test_ids_file) as f:
        for line in f:
            issues[line.strip()] = []
    print(len(issues))
    with open(args.input_dataset_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        print(next(csv_reader))
        for row in csv_reader:
            if row[2] in issues:
                issues[row[2]].append(row[1])  # issue_link
                issues[row[2]].append(row[3])  # post

    with open(args.input_qa_file) as csvDataFile:
        csv_reader = csv.reader(csvDataFile, delimiter='\t')
        print(next(csv_reader))
        for row in csv_reader:
            if row[0] in issues:
                questions = []
                questions.append(row[1])
                questions.append(row[2])
                questions.append(row[3])
                questions.append(row[4])
                questions.append(row[5])
                questions.append(row[6])
                questions.append(row[7])
                questions.append(row[8])
                questions.append(row[9])
                questions.append(row[10])
                random.shuffle(questions)
                issues[row[0]].append(questions)
    print(len(issues))
    issue_ids = list(issues.keys())

    random.shuffle(issue_ids)
    count = 0
    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['issue_id', 'issue_link', 'post', 'q1', 'a1', 'q2', 'a2', 'q3', 'a3', 'q4', 'a4',
                         'q5', 'a5', 'q6', 'a6', 'q7', 'a7', 'q8', 'a8', 'q9', 'a9', 'q10', 'a10'])

    csv_file_2 = open(args.output_file_2, 'w')
    csv_writer_2 = csv.writer(csv_file_2)
    csv_writer_2.writerow(['issue_id', 'issue_link', 'post', 'q1', 'a1', 'q2', 'a2', 'q3', 'a3', 'q4', 'a4',
                         'q5', 'a5', 'q6', 'a6', 'q7', 'a7', 'q8', 'a8', 'q9', 'a9', 'q10', 'a10'])
    for issue_id in issue_ids:
        try:
            if count < 200:
                issue = issues[issue_id]
                output_row = []
                output_row.append(issue_id)
                output_row.append(issue[0])
                output_row.append(issue[1])
                for q in issue[2]:
                    output_row.append(q)
                    output_row.append('')
                csv_writer.writerow(output_row)
            else:
                issue = issues[issue_id]
                output_row = []
                output_row.append(issue_id)
                output_row.append(issue[0])
                output_row.append(issue[1])
                for q in issue[2]:
                    output_row.append(q)
                    output_row.append('')
                csv_writer_2.writerow(output_row)
            count = count + 1
            if count >= 400:
                break
        except:
            continue
    csv_file.close()
    csv_file_2.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--input_test_ids_file", type=str, default='/home/imranm3/Desktop/datasets_final_tag/test_ids.txt')
    argparser.add_argument("--input_qa_file", type=str, default='/home/imranm3/Desktop/datasets_final_tag/qa_data.tsv')
    argparser.add_argument("--input_dataset_file", type=str, default='/home/imranm3/Desktop/datasets_final_tag/dataset.csv')
    argparser.add_argument("--output_file", type=str, default='annotation_1.csv')
    argparser.add_argument("--output_file_2", type=str, default='annotation_2.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
