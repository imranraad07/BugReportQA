import argparse
import logging
import sys

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


# P_SR_HAVE_SEQUENCE
# Sequence of sentences using the verb "have"
# Example: I have FBML in a dialog. I have a div element that is position:absolute.
def setup_p_sr_have_sequence(doc, nlp):
    doc = doc.lower()
    have_count = 0
    match = False
    for sentence in nlp(doc).sents:
        sent = sentence.text.strip()
        if ("i have" in sent) or ("i\'ve" in sent):
            have_count = have_count + 1
        else:
            have_count = 0
        if have_count > 1:
            match = True
    return match


# S_SR_CODE_REF
def setup_s_sr_code_ref(doc):
    doc = doc.lower()
    match = re.search(
        '.*(code snippet|sample example|live example|test case)(.+?)(attached|below|provided|here|enclosed|following)(.+?)',
        doc)
    if match is not None:
        return True
    return False


# S_SR_WHEN_AFTER
def setup_s_sr_when_after(doc):
    doc = doc.lower()
    match = re.search(
        '^(when|if) (.+?) after (.+?)', doc)
    if match is not None:
        return True
    return False


if __name__ == '__main__':
    print(sys.argv)

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--repo_csv", type=str,
                           default='../data/datasets/dataset.csv')
    argparser.add_argument("--output_csv", type=str,
                           default='../data/bug_reports/github_data_s2r.csv')
    args = argparser.parse_args()

    issues = pd.read_csv(args.repo_csv)
    print("Total issues:", len(issues["post"]))

    nlp = spacy.load("en_core_web_sm")

    issue_matches = []
    s2r_list = []
    count = 0
    for index, issue in issues.iterrows():
        # paragraph matching steps to reproduce
        paragraph = (issue["post"]).lower()
        matched = False
        s2r_sent = ""
        if setup_p_sr_labeled_list(paragraph) or setup_p_sr_labeled_paragraph(paragraph) \
                or setup_p_sr_have_sequence(paragraph, nlp):
            matched = True
            s2r_sent = paragraph

        # sentence matching steps to reproduce
        if not matched:
            s2r_sent = ""
            for sentence in nlp(issue["post"]).sents:
                sent = sentence.text.strip()
                if setup_s_sr_code_ref(sent) or setup_s_sr_when_after(sent):
                    matched = True
                    s2r_sent = s2r_sent + " " + sent

        if matched:
            count = count + 1
            print(count, index, issue["issue_link"])
            s2r_list.append(s2r_sent)
        issue_matches.append(matched)

    print(count)
    s2r_issues = issues[issue_matches]
    print(len(s2r_issues), len(s2r_list))
    assert (len(s2r_issues) == len(s2r_list))
    s2r_issues = s2r_issues.assign(STR=s2r_list)
    s2r_issues.to_csv(args.output_csv, index=False)
