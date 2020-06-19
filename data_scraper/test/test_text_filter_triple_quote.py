import unittest

from data_scraper.github_text_filter import remove_triple_quotes as filter


class TestRecPatterns(unittest.TestCase):

    def test_triple_quotes(self):
        br = "the triple quote ```something``` removal is\n ```something``` correct. ```is it?```"
        print(br)
        text = filter(br)
        text = " ".join(text.split())
        print(text)
        self.assertEqual("the triple quote removal is correct.".strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
