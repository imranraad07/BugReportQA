import spacy
from spacy.lang.en import English
from spacy.matcher.matcher import Matcher

import pattern_classification.steps_to_reproduce_rule as s2r
import unittest


class TestRecPatterns(unittest.TestCase):

    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        return unittest.TestCase.setUp(self)

    # def test_basecase_s_sr_when_after(self):
    #     s2r.setup_s_sr_when_after(self.matcher)
    #     sent = "when I press the  publish  button on the facebook interface after entering some text about a post_ I get an empty window with only an  x  in the upper right-hand corner and the app just sits there."
    #     sent_nlp = self.nlp(sent)
    #     for token in sent_nlp:
    #         print(token.pos_, token.text, token.lemma_)
    #     matches = self.matcher(sent_nlp)
    #     self.assertTrue(len(matches) == 1)


    def test_basecase_p_s2r_labeled_list(self):
        doc = "The problem occurs when I tried this. steps to reproduce:  1. Create a bar chart" \
              " 2. Open chart builder " \
              " 3. click filters step 4. preview"
        self.assertTrue(s2r.setup_p_sr_labeled_list(doc) == True)

        doc = "steps to reproduce:  1. Create a bar chart" \
              " 2. Open chart builder " \
              " 3. click filters step 4. preview"
        self.assertTrue(s2r.setup_p_sr_labeled_list(doc) == True)

        doc = "The problem occurs when I tried this. steps to reproduce: click, open, edit, delete"
        self.assertTrue(s2r.setup_p_sr_labeled_list(doc) == False)

    def test_basecase_p_s2r_labeled_paragraph(self):
        doc = "The problem occurs when I tried this. steps to reproduce:  Create a bar chart." \
              " Open chart builder. " \
              " click filters. preview."
        self.assertTrue(s2r.setup_p_sr_labeled_paragraph(doc) == True)

        doc = "The problem occurs when I tried this. steps to reproduce: click, open, edit, delete"
        self.assertTrue(s2r.setup_p_sr_labeled_paragraph(doc) == False)

        doc = "steps to recreate the problem:  Create a bar chart." \
              " Open chart builder. " \
              " click filters. preview."
        self.assertTrue(s2r.setup_p_sr_labeled_paragraph(doc) == True)


if __name__ == "__main__":
    unittest.main()
