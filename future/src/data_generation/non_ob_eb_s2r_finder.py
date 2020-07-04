import argparse
import csv
import sys
import pattern_classification.observed_behavior_rule as ob_rule
import pattern_classification.steps_to_reproduce_rule as s2r_rule
import pattern_classification.expected_behavior_rule as eb_rule
import spacy
from spacy.matcher import Matcher


class FindOB_S2R_EB(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.ob = FindOB(self.nlp, threshold)
        self.s2r = FindS2R(self.nlp)
        self.eb = FindEB(self.nlp)

    def is_eligible(self, post):
        post = post.strip()
        if self.ob.is_ob(post) is True or self.s2r.is_s2r(post) is True or self.eb.is_eb(post) is True:
            return False
        return True


class FindOB(object):

    def __init__(self, nlp, threshold=0.0001):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.threshold = threshold
        ob_rule.setup_s_ob_neg_aux_verb(self.matcher)
        ob_rule.setup_s_ob_verb_error(self.matcher)
        ob_rule.setup_s_ob_neg_verb(self.matcher)
        ob_rule.setup_s_ob_but(self.matcher)
        ob_rule.setup_s_ob_cond_pos(self.matcher)

    # return true if match
    def is_ob(self, text):
        flag = False
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
                    flag = True
        # print(flag, ob_str)
        return flag


class FindEB(object):

    def __init__(self, nlp, threshold=0.0001):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.threshold = threshold
        eb_rule.setup_s_eb_exp_behavior(self.matcher)
        eb_rule.setup_s_eb_expected(self.matcher)
        eb_rule.setup_s_eb_instead_of_expected_behavior(self.matcher)
        eb_rule.setup_s_eb_should(self.matcher)
        eb_rule.setup_s_eb_would_be(self.matcher)

    # return true if match
    def is_eb(self, text):
        flag = False
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
                    flag = True
        # print(flag, eb_str)
        return flag


class FindS2R(object):

    def __init__(self, nlp):
        self.nlp = nlp

    # return true if match
    def is_s2r(self, text):
        _, s2r_text = s2r_rule.match_sr(text, join_by='\n')
        flag = True
        if s2r_text is None:
            s2r_text = ''
            flag = False
        # print(flag, s2r_text)
        return flag


def main(args):
    findOB_S2R_EB = FindOB_S2R_EB()
    count = 0
    filtered = 0

    csv_file = open(args.output_file, 'w')
    csv_writer = csv.writer(csv_file)

    with open(args.input_file) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        # repo, issue_link, issue_id, post, question, answer
        header = next(csv_reader)
        csv_writer.writerow(header)
        for row in csv_reader:
            count = count + 1
            if findOB_S2R_EB.is_eligible(row[3]):
                csv_writer.writerow(row)
            else:
                filtered = filtered + 1
            if count % 10 is 0:
                print(count, filtered)
    csv_file.close()
    print(count)
    print(filtered)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--input_file", type=str,
                           default='../../../data/datasets/github/dataset_filtered.csv')
    argparser.add_argument("--output_file", type=str,
                           default='../../../data/datasets/github/dataset.csv')

    csv.field_size_limit(sys.maxsize)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
