import csv

import observed_behavior_rule as ob_rule
import steps_to_reproduce_rule as s2r_rule
import expected_behavior_rule as eb_rule
import spacy
import pandas as pd
from spacy.matcher import Matcher

eb_calculated = {}
ob_calculated = {}
s2r_calculated = {}
post_utility = {}


class Calculator(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.ob = CalculatorOB(self.nlp, threshold)
        self.eb = CalculatorEB(self.nlp, threshold)
        self.s2r = CalculatorS2R(self.nlp)

    def utility(self, postid, answer):
        ob_text = ob_calculated[postid]
        ob_sents = 0 if len(ob_text) == 0 else len(ob_text.split('\n'))
        # print('OB: {0}'.format(ob_text))

        eb_text = eb_calculated[postid]
        eb_sents = 0 if len(eb_text) == 0 else len(eb_text.split('\n'))
        # print('EB: {0}'.format(eb_text))

        s2r_text = s2r_calculated[postid]
        s2r_sents = 0 if len(s2r_text) == 0 else len(s2r_text.split('\n'))
        # print('S2R: {0}'.format(s2r_text))

        answer_sents = len(list(self.nlp(answer).sents))
        utility = (ob_sents + s2r_sents + eb_sents) / float(answer_sents)
        # print('Utility: {0} + {1} + {2} / {3} = {4}\n\n'.format(ob_sents, eb_sents, s2r_sents, answer_sents, utility))
        return max(min(utility, 1.0), self.threshold)


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


def compute_utilities(answer_ob_eb_s2r_csv, data_avg_tsv, post_tsv, qa_tsv, qa_post_ids, utility_tsv):
    calculator = Calculator()

    print("collecting similar question posts...")
    similar_post_set_avg = {}
    with open(data_avg_tsv) as csvDataFile:
        csv_reader = csv.reader(csvDataFile, delimiter='\t')
        row = next(csv_reader)
        print(row)
        print(len(row))
        for row in csv_reader:
            record = []
            record.append(row[1])
            record.append(row[4])
            record.append(row[7])
            record.append(row[10])
            record.append(row[13])
            record.append(row[16])
            record.append(row[19])
            record.append(row[22])
            record.append(row[25])
            record.append(row[28])
            similar_post_set_avg[row[0]] = record
    print("similar_post_set_avg len", len(similar_post_set_avg))
    print("done")

    print("collecting answer ob eb s2r...")
    with open(answer_ob_eb_s2r_csv) as csvDataFile:
        csv_reader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        # postid, answer, ob_text, eb_text, s2r_text
        next(csv_reader)
        for row in csv_reader:
            ob_calculated[row[0]] = row[2]
            eb_calculated[row[0]] = row[3]
            s2r_calculated[row[0]] = row[4]
            post_utility[row[0]] = calculator.utility(row[0], row[1])
            # print(row[0])
    print("done")
    print("ob len:", len(ob_calculated))
    print("eb len:", len(eb_calculated))
    print("s2r len:", len(s2r_calculated))
    # print("post utility len:", len(post_utility))
    # print(post_utility)

    avg_utility = {}
    for item in post_utility:
        # print("avg utility", item)
        if item in similar_post_set_avg:
            utility_a = 0.0
            posts = similar_post_set_avg[item]
            for post in posts:
                utility_a = utility_a + float(post_utility[post])
                # print("    --", utility_a)
            utility_a = utility_a / 10
        else:
            utility_a = post_utility[post]
        avg_utility[item] = utility_a
        print(row[0], post_utility[row[0]], utility_a)

    print("computing utilities...")
    posts = pd.read_csv(post_tsv, sep='\t')
    qa = pd.read_csv(qa_tsv, sep='\t')

    utility_data = {'postids': list()}
    for i in range(1, 11):
        utility_data['p_a{0}'.format(i)] = list()

    for idx, row in posts.iterrows():
        # print('Row {0}/{1}'.format(idx + 1, len(posts)))
        postid = row['postid']
        utility_data['postids'].append(postid)
        answer_post_ids = qa_post_ids[postid]
        utility_record = []
        utility_record.append(postid)
        for i in range(1, 11):
            answer = qa.iloc[idx]['a' + str(i)]
            utility = avg_utility[answer_post_ids[i - 1]]
            utility_data['p_a{0}'.format(i)].append(utility)
            utility_record.append(utility)

    df = pd.DataFrame(utility_data)
    df.to_csv(utility_tsv, index=False, sep='\t')
