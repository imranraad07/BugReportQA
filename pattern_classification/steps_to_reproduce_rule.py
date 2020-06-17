import argparse
import re
import sys

import pandas as pd
import spacy


# s2r_labels = ["how to reproduce", "what i have tried", "to replicate", "steps to reproduce", "steps to recreate"]


def match_p_sr_labeled_list(doc):
    doc = doc.lower()
    match = re.search(
        '.*(how to reproduce|what i have tried|to replicate|steps to reproduce|steps to recreate)(.+?)((\d+\.\s.*\n?)+s)(.+?)',
        doc)
    if match is not None:
        return True
    return False


def match_p_sr_labeled_paragraph(doc):
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
def match_p_sr_have_sequence(doc, nlp):
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
def match_s_sr_code_ref(doc):
    doc = doc.lower()
    match = re.search(
        '.*(code snippet|sample example|live example|test case)(.+?)(attached|below|provided|here|enclosed|following)(.+?)',
        doc)
    if match is not None:
        return True
    return False


# S_SR_WHEN_AFTER
def match_s_sr_when_after(doc):
    doc = doc.lower()
    match = re.search(
        '^(when|if) (.+?) after (.+?)', doc)
    if match is not None:
        return True
    return False


def match_sr(text, join_by=' '):
    nlp = spacy.load("en_core_web_sm")
    text = text.lower()
    matched = False
    s2r_sent = ""

    # paragraph level matching steps to reproduce
    if match_p_sr_labeled_list(text) or match_p_sr_labeled_paragraph(text) \
            or match_p_sr_have_sequence(text, nlp):
        matched = True
        s2r_sent = text

    # sentence level matching steps to reproduce
    if not matched:
        s2r_sent = ""
        for sentence in nlp(text).sents:
            sent = sentence.text.strip()
            if match_s_sr_code_ref(sent) or match_s_sr_when_after(sent):
                matched = True
                s2r_sent = s2r_sent + join_by + sent

    if matched:
        return matched, s2r_sent
    return matched, None


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
        s2r = match_sr(issue["post"])
        print(s2r)
        if s2r[0] is True:
            count = count + 1
            print(count, index, issue["issue_link"])
            s2r_list.append(s2r[1])
        issue_matches.append(s2r[0])

    print("Total s2r issues:", count)
    s2r_issues = issues[issue_matches]
    print(len(s2r_issues), len(s2r_list))
    assert (len(s2r_issues) == len(s2r_list))
    s2r_issues = s2r_issues.assign(STR=s2r_list)
    s2r_issues.to_csv(args.output_csv, index=False)
