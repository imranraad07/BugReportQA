import argparse
import csv
import math
import operator
import sys


def run(args):
    issue_post_mapping = {}
    with open(args.github_csv) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        next(csvReader)
        for row in csvReader:
            issue_post_mapping[row[2]] = row[3]
    print("post_data", len(issue_post_mapping.keys()))

    issue_qa_mapping = {}
    with open(args.qa_data_tsv) as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile), delimiter='\t')
        next(csvReader)
        for row in csvReader:
            issue_qa_mapping[row[0]] = (
                row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
    print("qa_data", len(issue_qa_mapping.keys()))

    test_ids = open(args.test_ids, 'r')
    Lines = test_ids.readline
    test_ids = []
    for line in Lines:
        test_ids.append(line.strip())

    epoch_test_out_list = [args.epoch0, args.epoch13, args.epch19]
    test_csv_out_list = [args.test_predictions_epoch0_csv, args.test_predictions_epoch13_csv,
                         args.test_predictions_epoch19_csv]

    idx = 0
    for out_file in epoch_test_out_list:
        csv_file = open(test_csv_out_list[idx], 'w')
        idx = idx + 1
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['issueid', 'post' 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'correct question'])

        epoch_out = open(out_file, 'r')
        Lines = epoch_out.readlines()
        row_cnt = 0
        for line in Lines:
            try:
                line_split = line.split()
                cnt = 0
                predic_dict = {}
                correct_answer = ""
                post_id = ""
                nan_found = False
                for text in line_split:
                    if cnt > 0:
                        predic_dict[cnt] = (float(text), issue_qa_mapping[post_id][cnt - 1])
                        if math.isnan(float(text)) is True:
                            nan_found = True
                    if cnt == 1:
                        correct_answer = issue_qa_mapping[post_id][0]
                    if cnt == 0:
                        post_id = text.strip()
                        post_id = post_id[1:]
                        post_id = post_id[:len(post_id) - 2]
                    cnt = cnt + 1
                # if nan_found is True:
                #     continue
                predic_dict = sorted(predic_dict.items(), key=operator.itemgetter(1), reverse=True)
                row_cnt = row_cnt + 1
                row_val = []
                row_val.append(post_id)
                row_val.append(issue_post_mapping[post_id])
                for item in predic_dict:
                    row_val.append(item[1][1])
                row_val.append(correct_answer)

                csv_writer.writerow(row_val)
            except Exception as e:
                continue
        csv_file.close()


if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data_dir", type=str, default='github')
    argparser.add_argument("--qa_data_tsv", type=str)
    argparser.add_argument("--github_csv", type=str)
    argparser.add_argument("--train_ids", type=str)
    argparser.add_argument("--epoch0", type=str)
    argparser.add_argument("--epoch13", type=str)
    argparser.add_argument("--epoch19", type=str)
    argparser.add_argument("--test_predictions_epoch0_csv", type=str)
    argparser.add_argument("--test_predictions_epoch13_csv", type=str)
    argparser.add_argument("--test_predictions_epoch19_csv", type=str)
    args = argparser.parse_args()
    print
    args
    print
    run(args)
