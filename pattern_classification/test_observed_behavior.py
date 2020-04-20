import observed_behavior_rule as obr
import spacy
from spacy.matcher import Matcher
import unittest


class TestRecPatterns(unittest.TestCase):

    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        return unittest.TestCase.setUp(self)

    def test_basecase_s_ob_neg_aux_verb(self):
        obr.setup_s_ob_neg_aux_verb(self.matcher)
        sent_nlp = self.nlp("The icon did not change to an hourglass")
        # for token in sent_nlp:
        #     print(token.pos_,token.text,token.lemma_)
        matches = self.matcher(sent_nlp)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(sent_nlp[start:end].text,"did not change")

    def test_basecase_s_ob_verb_error(self):
        obr.setup_s_ob_verb_error(self.matcher)
        sent_nlp = self.nlp("VirtualBOx GUI gives this error:")
        # for token in sent_nlp:
        #     print(token.pos_,token.text,token.lemma_)
        matches = self.matcher(sent_nlp)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(sent_nlp[start:end].text,"gives this error")

    def test_basecase_s_ob_neg_verb(self):
        obr.setup_s_ob_neg_verb(self.matcher)
        sent_nlp = self.nlp("Writer hangs on opening some doc, docx or rtf files")
        # for token in sent_nlp:
        #     print(token.pos_,token.text,token.lemma_)
        matches = self.matcher(sent_nlp)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(sent_nlp[start:end].text,"hangs")

if __name__ == "__main__":
    unittest.main()