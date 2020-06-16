import observed_behavior_rule as ob_rule
import steps_to_reproduce_rule as s2r_rule
import spacy
from spacy.matcher import Matcher
from difflib import Differ


class Calculator(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold
        self.ob = CalculatorOB(self.nlp, threshold)
        self.s2r = CalculatorS2R(self.nlp)

    def utility(self, answer, post):
        answer = answer.strip()
        # nlp.sents treats * as sentence separator
        answer = answer.replace('*', ' ').replace('...', '.')
        print('Answer: {0}'.format(answer))
        ob_text = self.ob.get_ob(answer).strip()
        ob_sents = 0 if len(ob_text) == 0 else len(ob_text.split('\n'))
        print('OB: {0}'.format(ob_text))
        s2r_text = self.s2r.get_s2r(answer).strip()
        s2r_sents = 0 if len(s2r_text) == 0 else len(s2r_text.split('\n'))
        print('S2R: {0}'.format(s2r_text))
        answer_sents = len(list(self.nlp(answer).sents))
        utility = (ob_sents + s2r_sents) / float(answer_sents)
        print('Utility: {0} + {1} / {2} = {3}\n\n'.format(ob_sents, s2r_sents, answer_sents, utility))
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
