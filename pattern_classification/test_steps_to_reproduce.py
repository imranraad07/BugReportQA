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

    def test_basecase_s_sr_code_ref(self):
        doc = "the code snippet below opens a shell that goes across multiple monitors when there are multiple monitors installed."
        self.assertTrue(s2r.setup_s_sr_code_ref(doc) == True)

    def test_basecase_p_sr_have_sequence(self):
        doc = "I have FBML in a dialog. I have a div element that is position:absolute."
        self.assertTrue(s2r.setup_p_sr_have_sequence(doc, self.nlp) == True)

    def test_basecase_s_sr_when_after(self):
        doc = "When I press the  Publish  button on the facebook interface after entering some text about a post_ I get an empty window with only an  X  in the upper right-hand corner and the app just sits there."
        self.assertTrue(s2r.setup_s_sr_when_after(doc) == True)
        doc = "When I press the  Publish  button on the facebook interface, no exception occurs"
        self.assertTrue(s2r.setup_s_sr_when_after(doc) == False)


if __name__ == "__main__":
    unittest.main()
