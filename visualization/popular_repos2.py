import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


# golang/go, kubernetes/kubernetes, rust-lang/rust, ansible/ansible, microsoft/TypeScript, elastic/elasticsearch,
# godotengine/godot, saltstack/salt, angular/angular, moby/moby

def main(input_file):
    df_init = pd.read_csv(input_file, header=None)
    df = df_init.transpose()

    # put first row as header
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header
    df = df.astype('int32')

    # colors = ['brown', 'pink', 'black', 'violet', 'red', 'gray', 'indigo', 'orange', 'green', 'blue']
    # boxplot = df.boxplot()
    fig, ax = plt.subplots()
    ax.set_yticklabels(df.columns.tolist(), fontsize=14)
    # ax.tick_params(axis='y', which='major', pad=15)
    ax.set_xlabel('created issues per day (year=2019)', fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.boxplot(df.values, vert=False, showfliers=False)
    fig.savefig("popular_repos.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str, default='day_issue_count.csv')
    args = argparser.parse_args()
    main(args.input_file)
