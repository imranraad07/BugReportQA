import argparse
import csv
import sys
import time
from datetime import date, timedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# golang/go, kubernetes/kubernetes, rust-lang/rust, ansible/ansible, microsoft/TypeScript, elastic/elasticsearch,
# godotengine/godot, saltstack/salt, angular/angular, moby/moby

def main(args):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cdays_in_month = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    months_name = ['Jan-19', 'Feb-19', 'Mar-19', 'Apr-19', 'May-19', 'June-19', 'July-19',
                   'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19']
    months = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    plt.style.use('fivethirtyeight')
    bug_reports_month = []
    with open(args.input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        gc = 1
        for row in csv_reader:
            label = ''
            count = 0
            br_count = 0
            month_count = 1
            for item in row:
                if count == 0:
                    label = item
                else:
                    if count in cdays_in_month:
                        bug_reports_month.append([months[month_count - 1], label, br_count])
                        month_count = month_count + 1
                        br_count = 0
                    br_count = br_count + int(item)
                count = count + 1

    df = pd.DataFrame(bug_reports_month,
                      columns=['months', 'repos', 'number of issues'])

    colors = ['brown', 'pink', 'black', 'violet', 'red', 'gray', 'indigo', 'orange', 'green', 'blue']
    df.pivot('months', 'repos', 'number of issues').plot(kind='bar', width=0.80, color=colors)
    plt.xticks(months, months_name, rotation=0)
    # plt.title('Bug Report Traffic Bursty - 2019')
    plt.xlabel('Month')
    plt.ylabel('Number of Created Issues')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--input_file", type=str, default='day_issue_count.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
