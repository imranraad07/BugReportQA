import os
import io
import sys
import time
import logging
import requests
from gensim.utils import to_unicode, to_utf8
from lxml import etree
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
import csv

logger = logging.getLogger('jira')
os.environ['TZ'] = 'UTC'


def download_jira_bugs(output, repo_name):
    count = 0
    url_base = 'https://issues.apache.org/jira/si/jira.issueviews:issue-xml/%s/%s.xml'
    mkdir(output)

    p = etree.XMLParser()
    hp = etree.HTMLParser()

    bugid = 0
    fail_attempts_cnt = 0

    while fail_attempts_cnt < 50:
        if bugid == 500:
            break
        bugid += 1
        logger.info("Fetching bugid %s", bugid)
        fname = repo_name.upper() + '-' + str(bugid)

        print(url_base % (fname, fname));
        r = try_request(url_base % (fname, fname))
        r = to_unicode(r.text)

        try:
            tree = etree.parse(io.StringIO(r), p)
        except etree.XMLSyntaxError:
            logger.error("Error in XML: {0} - {1}".format(repo_name, bugid))
            fail_attempts_cnt += 1
            continue

        root = tree.getroot()

        type = root.find('channel').find('item').find('type').text

        html = root.find('channel').find('item').find('description').text
        summary = root.find('channel').find('item').find('summary').text
        summary = to_unicode(summary)

        fix_version = ["v" + x.text for x in root.findall('.//fixVersion')]
        affected_versions = ["v" + x.text for x in root.find('channel').find('item').findall('.//version')]

        created_time = root.find('channel').find('item').findall('.//created')
        assert len(created_time) == 1
        created_time = get_time(created_time[0].text)

        htree = etree.parse(io.StringIO(html), hp)
        if htree.getroot() is not None:
            desc = ''.join(htree.getroot().itertext())
            desc = to_unicode(desc)

            comments = root.find('channel').find('item').find('comments')
            if comments is not None:
                print(len(list(comments)));
                for comment in list(comments):
                    id = comment.get('id')
                    text = BeautifulSoup(comment.text).get_text().replace('\n', ' ')
                    # text = BeautifulSoup(comment.text).get_text()
                    # print(sent_tokenize(text))
                    for sentence in sent_tokenize(text):
                        if check(sentence):
                            print(count, ' ', sentence)
                            count = count + 1
                            sw = csv.writer(open('{0}/data_zookeeper.csv'.format(output), 'a'))
                            sw.writerow([
                                'ZOOKEEPER-{0}'.format(bugid),
                                '{0}'.format(sentence)
                            ])

                    author = comment.get('author')
                    time = get_time(comment.get('created'))


def check(sentence):
    start_words = ['who', 'what', 'when', 'where', 'why', 'which', 'how']
    if sentence.endswith('?'):
        return True
    for word in start_words:
        if sentence.startswith(word):
            return True
    return False


def try_request(url, n=100):
    try:
        return requests.get(url)
    except:
        time.sleep(600 / n)
        try_request(url, n - 1)


def get_time(date_str):
    date_str = date_str.split(',')[1].strip()
    data_format = '%d %b %Y %H:%M:%S +0000'
    date = time.strptime(date_str, data_format)
    # to epoch time - easier to compare
    return int(time.mktime(date))


def mkdir(d):
    # exception handling mkdir -p
    try:
        os.makedirs(d)
    except os.error as e:
        if 17 == e.errno:
            # the directory already exists
            pass
        else:
            print('Failed to create "%s" directory!' % d)
            sys.exit(e.errno)


if __name__ == '__main__':
    download_jira_bugs('results', 'zookeeper')
