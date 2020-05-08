import unittest
import github_scraper


class TestGithubScraper(unittest.TestCase):
    def test_something(self):
        edits = github_scraper.get_edits("https://github.com/imranraad07/BugReportQA/", 31)
        self.assertEqual(len(edits), 3)
        self.assertEqual(edits[0][0], 'Now modify description')
        self.assertEqual(edits[1][0], 'Add description')
        self.assertEqual(edits[2][0], '')
        self.assertEqual(edits[0][1], '2020-05-06T19:54:23Z')


if __name__ == '__main__':
    unittest.main()
