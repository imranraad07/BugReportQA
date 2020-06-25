import unittest

from future.src.data_generation.github_text_filter import remove_triple_quotes as filter


class TestRecPatterns(unittest.TestCase):

    def test_triple_quotes(self):
        br = "the triple quote ```something``` removal is\n ```something``` correct. ```is it?```"
        text = filter(br)
        text = " ".join(text.split())
        self.assertEqual("the triple quote removal is correct.".strip(), text.strip())


if __name__ == "__main__":
    unittest.main()
