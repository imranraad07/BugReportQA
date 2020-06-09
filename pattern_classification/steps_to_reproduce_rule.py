import logging

import spacy
import pandas as pd
from spacy.matcher import Matcher
import re


# s2r_labels = ["how to reproduce", "what i have tried", "to replicate", "steps to reproduce", "steps to recreate"]


def setup_p_sr_labeled_list(doc):
    doc = doc.lower()
    match = re.search(
        '.*(how to reproduce|what i have tried|to replicate|steps to reproduce|steps to recreate)(.+?)((\d+\.\s.*\n?)+s)(.+?)',
        doc)
    if match is not None:
        return True
    return False


def setup_p_sr_labeled_paragraph(doc):
    doc = doc.lower()
    match = re.search(
        '.*(how to reproduce|what i have tried|to replicate|steps to reproduce|steps to recreate)(.+?)((.)+s)(.+?)',
        doc)
    if match is not None:
        return True
    return False


# S_SR_CODE_REF
def setup_s_sr_code_ref(doc):
    doc = doc.lower()
    match = re.search(
        '.*(code snippet|sample example|live example|test case)(.+?)(attached|below|provided|here|enclosed|following)(.+?)',
        doc)
    if match is not None:
        return True
    return False


if __name__ == '__main__':

    issues = pd.read_csv('../data/datasets/dataset.csv')

    print("Total issues:", len(issues["post"]))
    nlp = spacy.load("en_core_web_sm")
    # matcher = Matcher(nlp.vocab, validate=True)
    # setup_s_sr_code_ref(matcher)

    issue_matches = []
    s2r_list = []
    count = 0
    for index, issue in issues.iterrows():
        # paragraph matching steps to reproduce
        paragraph = (issue["post"]).lower()
        matched = False
        if setup_p_sr_labeled_list(paragraph) or setup_p_sr_labeled_paragraph(paragraph):
            matched = True
            s2r_list.append(paragraph)
            count = count + 1
            print(count, index, issue["issue_link"])
        # # sentence matching steps to reproduce
        if not matched:
            s2r_sent = ""
            for sentence in nlp(issue["post"]).sents:
                sent = sentence.text.strip()
                flag = setup_s_sr_code_ref(sent)
                if flag:
                    matched = True
                    s2r_sent = s2r_sent + " " + sent
            if matched:
                s2r_list.append(s2r_sent)
                count = count + 1
                print(count, index, issue["issue_link"])
        issue_matches.append(matched)

    print(count)
    s2r_issues = issues[issue_matches]
    assert (len(s2r_issues) == len(s2r_list))
    s2r_issues = s2r_issues.assign(STR=s2r_list)
    s2r_issues.to_csv("../data/bug_reports/github_data_s2r.csv", index=False)
