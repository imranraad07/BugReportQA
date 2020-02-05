import os
import io
import time
import logging
import requests
import corley.utils as utils
from gensim.utils import to_unicode, to_utf8
from lxml import etree
from bs4 import BeautifulSoup

logger = logging.getLogger('jira')
os.environ['TZ'] = 'UTC'


def download_jira_bugs(output, repo_name):
    url_base = 'https://issues.apache.org/jira/si/jira.issueviews:issue-xml/%s/%s.xml'
    utils.mkdir(output)

    p = etree.XMLParser()
    hp = etree.HTMLParser()

    bugid = 0
    fail_attempts_cnt = 0

    while fail_attempts_cnt < 50:
        bugid += 1
        logger.info("Fetching bugid %s", bugid)
        fname = repo_name.upper() + '-' + str(bugid)

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
        desc = ''.join(htree.getroot().itertext())
        desc = to_unicode(desc)

        comments = root.find('channel').find('item').find('comments')
        for comment in list(comments):
            id = comment.get('id')
            text = BeautifulSoup(comment.text).get_text().repalce('\n', ' ')
            author = comment.get('author')
            time = get_time(comment.get('created'))


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


class Issue:

    def __init__(self, short_desc, long_desc, type, comments):
        self.short_desc = short_desc
        self.long_desc = long_desc
        self.type = type
        self.comments = comments


class Comment:

    def __init__(self, text, time, author):
        self.text = text
        self.time = time
        self.author = author


if __name__ == '__main__':
    download_jira_bugs('test', 'bookkeeper')
