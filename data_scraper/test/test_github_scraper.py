import unittest
import github_scraper


class TestGithubScraper(unittest.TestCase):
    def test_something(self):
        edits = github_scraper.get_edits("https://github.com/imranraad07/BugReportQA/", 31)
        self.assertEqual(len(edits), 3)
        self.assertEqual(edits[0], 'Now modify description')
        self.assertEqual(edits[1], 'Add description')
        self.assertEqual(edits[2], '')


if __name__ == '__main__':
    unittest.main()
