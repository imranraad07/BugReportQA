import argparse
import csv
import sys

import pattern_classification.observed_behavior_rule as ob_rule
import pattern_classification.steps_to_reproduce_rule as s2r_rule
import pattern_classification.expected_behavior_rule as eb_rule
import spacy
import pandas as pd
from spacy.matcher import Matcher


class Calculator(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.ob = CalculatorOB(self.nlp, threshold)
        self.eb = CalculatorEB(self.nlp, threshold)
        self.s2r = CalculatorS2R(self.nlp)

    def utility(self, answer):
        answer = answer.strip()
        answer = answer.replace('*', ' ').replace('...', '.')
        ob_text = self.ob.get_ob(answer).strip()
        eb_text = self.eb.get_eb(answer).strip()
        s2r_text = self.s2r.get_s2r(answer).strip()
        return ob_text, eb_text, s2r_text


class CalculatorOB(object):

    def __init__(self, nlp, threshold=0.0001):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.threshold = threshold
        ob_rule.setup_s_ob_neg_aux_verb(self.matcher)
        ob_rule.setup_s_ob_verb_error(self.matcher)
        ob_rule.setup_s_ob_neg_verb(self.matcher)
        ob_rule.setup_s_ob_but(self.matcher)
        ob_rule.setup_s_ob_cond_pos(self.matcher)

    def get_ob(self, text):
        ob_str = ''
        for sentence in self.nlp(text).sents:
            sent = sentence.text.strip()
            if sent.startswith('>') or sent.endswith('?'):
                continue
            else:
                sent_nlp = self.nlp(sent)
                matches = self.matcher(sent_nlp)
                if len(matches) >= 1:
                    ob_str += sent + '\n'
        return ob_str


class CalculatorEB(object):

    def __init__(self, nlp, threshold=0.0001):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.threshold = threshold
        eb_rule.setup_s_eb_exp_behavior(self.matcher)
        eb_rule.setup_s_eb_expected(self.matcher)
        eb_rule.setup_s_eb_instead_of_expected_behavior(self.matcher)
        eb_rule.setup_s_eb_should(self.matcher)
        eb_rule.setup_s_eb_would_be(self.matcher)

    def get_eb(self, text):
        eb_str = ''
        for sentence in self.nlp(text).sents:
            sent = sentence.text.strip()
            if sent.startswith('>') or sent.endswith('?'):
                continue
            else:
                sent_nlp = self.nlp(sent)
                matches = self.matcher(sent_nlp)
                if len(matches) >= 1:
                    eb_str += sent + '\n'
        return eb_str


class CalculatorS2R(object):

    def __init__(self, nlp):
        self.nlp = nlp

    def get_s2r(self, text):
        _, s2r_text = s2r_rule.match_sr(text, join_by='\n')
        if s2r_text is None:
            return ''
        return s2r_text


def main(args):
    calculator = Calculator()
    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)
    output_record = ['postid', 'answer', 'ob_text', 'eb_text', 's2r_text']
    csv_writer.writerow(output_record)
    with open(args.input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        # repo, issue_link, issue_id, post, question, answer
        header = next(csv_reader)
        idx = 0
        for row in csv_reader:
            answer = row[5]
            answer = answer.strip()
            answer = answer.replace('*', ' ').replace('...', '.')

            ret = calculator.utility(answer)
            output_record = []
            output_record.append(row[2])
            output_record.append(row[5])
            output_record.append(ret[0])
            output_record.append(ret[1])
            output_record.append(ret[2])
            # print(output_record)
            print(idx, row[2])
            idx = idx + 1
            csv_writer.writerow(output_record)
    csv_file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str,
                           default='../data/datasets/datasets_final_tag/dataset.csv')
    argparser.add_argument("--output_file", type=str,
                           default='../data/datasets/datasets_final_tag/answer_ob_eb_s2r.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
