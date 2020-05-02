import unittest
from data_scraper.github_text_filter import filter_nontext as filter


class TestRecPatterns(unittest.TestCase):

    def test_block_quote(self):
        br = 'something \n' + \
             '> one ```code code code``` \n' + \
             '> two \n' + \
             'three'
        text = filter(br)
        self.assertEqual(text.strip(), "something three")

    def test_triple_quotes(self):
        br = 'something something ```code code code``` something'
        text = filter(br)
        self.assertEqual(text.strip(), "something something  something")

    def test_triple_quotes_nl(self):
        br = 'something something ```code \n code \n' + \
             'code```\nsomething'
        text = filter(br)
        self.assertEqual(text.strip(), "something something  something")

    def test_triple_quotes_eof(self):
        br = 'something something ```code \n code \n' + \
             'code\n' + \
             'code'
        text = filter(br)
        self.assertEqual(text.strip(), "something something")

    def test_remove_stacktrace1(self):
        with open('br_stacktrace1.txt', 'r') as f:
            text = ' '.join(f.readlines())
        text = filter(text)
        self.assertEqual(text.strip(),
                         "NoSuchMethodError `java.lang.NoSuchMethodError: No virtual method lambda$call$1(Lrx/CompletableSubscriber;Landroid/support/v4/view/ViewPropertyAnimatorCompat;)V in class Loxim/digital/rxanim/AnimateOnSubscribe; or its super classes (declaration of 'oxim.digital.rxanim.AnimateOnSubscribe' appears in /data/app/package-app-2/split_lib_dependencies_apk.apk) `")

    def test_remove_stacktrace2(self):
        with open('br_stacktrace2.txt', 'r') as f:
            text = ' '.join(f.readlines())
        text = filter(text)
        self.assertEqual(text.strip(),
                         'Incomplete database setup with PostGIS 3 Hi, I am trying to establish the connection to the 3DCityDB instance but there is an error I ca not identify. The citydb schema is already created on my database and I don\'tknow how to load the missing citydb_pkg as stated on the log. I would be happy with some help! By the way, if there is a best place to discuss issues related with 3D CityDB and/or CityGML, please, inform me. Thanks ![image](https://user-images.githubusercontent.com/31985605/68204969-20e57800-0004-11ea-88d0-100eb09fc914.png) `[19:20:23 INFO] Connecting to database profile \'Berlin model\'. [19:20:26 ERROR] Connection to database could not be established. [19:20:26 ERROR] Check the following stack trace for details: [19:20:26 ERROR] java.sql.SQLException: Failed to retrieve version information from the 3D City Database instance. ... 12 more ` "')

    def test_remove_stacktrace3(self):
        with open('br_stacktrace3.txt', 'r') as f:
            text = ' '.join(f.readlines())
        text = filter(text)
        with open('br_stacktrace3_correct.txt', 'r') as f:
            answer = ' '.join(f.readlines())
            answer = answer.replace('\n', ' ')
            # hack to get rid of multiple spaces in the text
            answer = ' '.join(answer.split())

        self.assertEqual(text.strip(), answer.strip())


if __name__ == "__main__":
    unittest.main()
