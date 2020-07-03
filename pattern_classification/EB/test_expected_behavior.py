import expected_behavior_rule as eb
import spacy
from spacy.matcher import Matcher
import unittest


class TestPatterns(unittest.TestCase):

    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        return unittest.TestCase.setUp(self)

    def test_s_eb_should(self):
        eb.setup_s_eb_should(self.matcher)
        doc = self.nlp("We should use the core project reference mechanism if possible")
        matches = self.matcher(doc)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(doc[start:end].text, "should")

    def test_s_eb_exp_behavior(self):
        eb.setup_s_eb_exp_behavior(self.matcher)
        doc = self.nlp("Expected Results:  displayed it s logo as usual")
        matches = self.matcher(doc)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(doc[start:end].text, "Expected Results")

    def test_s_instead_of_expected_behavior(self):
        eb.setup_s_eb_instead_of_expected_behavior(self.matcher)
        doc = self.nlp("fb:redirect tags are broken in IE_ instead of forwarding to:"
                       " http://apps.facebook.com/yesnomaybe/?age_range_seeking=4"
                       " it forwards to:"
                       " http://apps.facebook.com/yesnomaybe/[object]?age_range_seeking=4")
        matches = self.matcher(doc)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(doc[start:end].text, "instead of")

    def test_s_eb_would_be(self):
        eb.setup_s_eb_would_be(self.matcher)
        doc = self.nlp("It would be nice if I would have to enter all my installed JREs only once.")
        matches = self.matcher(doc)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(doc[start:end].text, "would be nice")

    def test_s_eb_expected(self):
        eb.setup_s_eb_expected(self.matcher)
        doc = self.nlp("The users expecting to press key UP/DOWN or Alt+DOWN to select an item. ")
        matches = self.matcher(doc)
        self.assertTrue(len(matches) == 1)
        start = matches[0][1]
        end = matches[0][2]
        self.assertEqual(doc[start:end].text, "expecting")


if __name__ == "__main__":
    unittest.main()
