import os


def join_files(dpath, out_fpath):
    lines = list()
    header = None

    for root, dirs, files in os.walk(dpath):
        for file in files:
            if 'github_data_20' in file:
                print('Processing {0}'.format(file))
                with open(os.path.join(root, file)) as f:
                    line = f.readline()
                    if header is None:
                        header = line
                    for line in f.readlines():
                        line = line.strip()
                        if len(line) > 0:
                            lines.append(line)
    print('Done')
    print('Save joined dataset to {0}'.format(out_fpath))
    with open(out_fpath, 'w') as f:
        f.write(header)
        for line in lines:
            f.write(line + '\n')
    print('Done')


if __name__ == '__main__':
    dpath = '/Users/ciborowskaa/VCU/Research/BugReportQA/data/bug_reports'
    out_fpath = os.path.join(dpath, 'github_dataset_partial.csv')
    join_files(dpath, out_fpath)
