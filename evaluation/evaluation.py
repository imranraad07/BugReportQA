import os
import argparse
import pandas as pd
import subprocess
import logging
import bleu
import mrr


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-fpath', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--ref-no', type=int, default=1)
    parser.add_argument('--meteor', type=str)
    return parser.parse_args()


def data4evaluation(in_data, out_refs, out_questions, ref_num):
    new_refs = open(out_refs, 'w')
    new_q = open(out_questions, 'w')
    df = pd.read_csv(in_data, sep=',', index_col=False)
    for index, row in df.iterrows():
        for i in range(1, ref_num + 1):
            new_refs.write(row['q' + str(i)] + '\n')
        new_q.write(row['correct_question'] + '\n')

    new_refs.close()
    new_q.close()

    print('Data prepared')


def run_meteor(ref_fpath, q_fpath, ref_no, output_path, meteor_fpath):
    command = u'java -Xmx2G -jar {0}.jar {1} {2} -l en -norm -r {3}'.format(meteor_fpath, q_fpath, ref_fpath, ref_no)
    commands = command.split(u' ')
    result = ''

    try:
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        while True:
            line = p.stdout.readline().decode()
            if line != '':
                result += line.rstrip() + '\n'
            else:
                break
    except Exception as e:
        print(e)

    with open(output_path, 'w') as f:
        f.write(result)


# java -Xmx2G -jar $METEOR/meteor-1.5.jar $QUESTIONS $REFS -l en -norm -r $REFS_NO > $OUTPUT_FPATH


def save(bleu_score, mrr_score, model_name, output_dir):
    with open(os.path.join(output_dir, model_name + '.mrr'), 'w') as f:
        f.write(str(mrr_score))
    with open(os.path.join(output_dir, model_name + '.bleu'), 'w') as f:
        f.write(str(bleu_score))


def process(args):
    results_fpath = args.results_fpath
    output_dir = args.output_dir
    ref_no = args.ref_no

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Processing {0}'.format(results_fpath))
    model_name = results_fpath.split('/')[-1].split('.')[0]

    ref_fpath = os.path.join(output_dir, model_name + '.refs')
    q_fpath = os.path.join(output_dir, model_name + '.questions')

    data4evaluation(results_fpath, ref_fpath, q_fpath, ref_no)
    bleu_score = bleu.compute(ref_fpath, q_fpath, ref_no)
    mrr_score = mrr.compute(results_fpath)
    run_meteor(ref_fpath, q_fpath, ref_no, os.path.join(output_dir, model_name + '.meteor'), args.meteor)

    save(bleu_score, mrr_score, model_name, output_dir)


if __name__ == '__main__':
    args = parse()
    process(args)
