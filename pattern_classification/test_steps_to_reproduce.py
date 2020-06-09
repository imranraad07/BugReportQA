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


if __name__ == "__main__":
    unittest.main()
