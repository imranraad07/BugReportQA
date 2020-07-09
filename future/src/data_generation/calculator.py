import observed_behavior_rule as ob_rule
import steps_to_reproduce_rule as s2r_rule
import expected_behavior_rule as eb_rule
import spacy
import pandas as pd
from spacy.matcher import Matcher
from difflib import Differ

eb_calculated = {}
ob_calculated = {}
s2r_calculated = {}


class Calculator(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.ob = CalculatorOB(self.nlp, threshold)
        self.eb = CalculatorEB(self.nlp, threshold)
        self.s2r = CalculatorS2R(self.nlp)

    def utility(self, postid, answer, post):
        answer = answer.strip()
        # nlp.sents treats * as sentence separator
        answer = answer.replace('*', ' ').replace('...', '.')
        print('Answer: {0}'.format(answer))

        if postid in ob_calculated:
            ob_text = ob_calculated[postid]
        else:
            ob_text = self.ob.get_ob(answer).strip()
            ob_calculated[postid] = ob_text
        ob_sents = 0 if len(ob_text) == 0 else len(ob_text.split('\n'))
        print('OB: {0}'.format(ob_text))

        if postid in eb_calculated:
            eb_text = eb_calculated[postid]
        else:
            eb_text = self.eb.get_eb(answer).strip()
            eb_calculated[postid] = eb_text
        eb_sents = 0 if len(eb_text) == 0 else len(eb_text.split('\n'))
        print('EB: {0}'.format(eb_text))

        if postid in s2r_calculated:
            s2r_text = s2r_calculated[postid]
        else:
            s2r_text = self.s2r.get_s2r(answer).strip()
            s2r_calculated[postid] = s2r_text
        s2r_sents = 0 if len(s2r_text) == 0 else len(s2r_text.split('\n'))
        print('S2R: {0}'.format(s2r_text))
        answer_sents = len(list(self.nlp(answer).sents))
        utility = (ob_sents + s2r_sents + eb_sents) / float(answer_sents)
        print('Utility: {0} + {1} + {2} / {3} = {4}\n\n'.format(ob_sents, eb_sents, s2r_sents, answer_sents, utility))
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


def get_diff(text_a, text_b):
    differ = Differ()
    diff_lines = differ.compare(text_a, text_b)
    diff = ' '.join([diff for diff in diff_lines if diff.startswith('+ ')]).replace('+ ', ' ')
    return diff


def compute_utilities(post_tsv, qa_tsv, utility_tsv):
    calculator = Calculator()
    posts = pd.read_csv(post_tsv, sep='\t')
    qa = pd.read_csv(qa_tsv, sep='\t')

    utility_data = {'postids': list()}
    for i in range(1, 11):
        utility_data['p_a{0}'.format(i)] = list()

    for idx, row in posts.iterrows():
        print('Row {0}/{1}'.format(idx + 1, len(posts)))
        postid = row['postid']
        print(postid)
        post = row['title'] + ' ' + row['post']
        utility_data['postids'].append(postid)
        for i in range(1, 11):
            answer = qa.iloc[idx]['a' + str(i)]
            utility = calculator.utility(postid, answer, post)

            utility_data['p_a{0}'.format(i)].append(utility)

    df = pd.DataFrame(utility_data)
    df.to_csv(utility_tsv, index=False, sep='\t')
