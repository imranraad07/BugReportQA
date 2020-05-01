import unittest
from data_scraper.github_text_filter import filter_nontext as filter


class TestRecPatterns(unittest.TestCase):

    def test_triple_quotes(self):
        br = 'something something ```code code code``` something'
        text = filter(br)
        self.assertEqual(text, "something something  something")


    def test_triple_quotes_nl(self):
        br = 'something something ```code \n code \n' + \
             'code```\nsomething'
        text = filter(br)
        self.assertEqual(text, "something something  something")

if __name__ == "__main__":
    unittest.main()