import spacy
import pandas as pd
from spacy.matcher import Matcher

should_terms = ["should", "shouldn't", "shall"]
expect_terms = ["expect", "expected", "expectation", "expecting"]
behavior_terms = ["behavior, results"]
instead_terms = ["instead of", "rather than"]
would_terms = ["would", "might"]

count_s_eb_should = 0
count_s_eb_exp_behavior = 0
count_s_eb_instead_of_expected_behavior = 0
count_s_eb_would_be = 0
count_s_eb_expected = 0


# S_EB_SHOULD
# Sentence using "should or shall"
# (I think/in my opinion) [non-conditional and non-neg aux verbs predicate]) [subject] should (not) [predicate]
def setup_s_eb_should(matcher):
    pattern = [{"LEMMA": {"IN": should_terms}}]
    matcher.add("S_EB_SHOULD", on_match_s_eb_should, pattern)


def on_match_s_eb_should(matcher, doc, i, matches):
    global count_s_eb_should
    count_s_eb_should = count_s_eb_should + 1


# S_EB_EXP_BEHAVIOR
# Sentence with explicit expected behavior keywords
# expected (behavior/result(s)): [sentence]

def setup_s_eb_exp_behavior(matcher):
    pattern = [{"LEMMA": {"IN": expect_terms}}, {"LEMMA": {"IN": behavior_terms}}]
    matcher.add("S_EB_EXP_BEHAVIOR", on_match_s_eb_exp_behavior, pattern)


def on_match_s_eb_exp_behavior(matcher, doc, i, matches):
    global count_s_eb_exp_behavior
    count_s_eb_exp_behavior = count_s_eb_exp_behavior + 1


# S_EB_INSTEAD_OF_EXPECTED_BEHAVIOR
# Use of "instead to" describe expected behavior
# [clause] instead of/rather than [predicate]
def setup_s_eb_instead_of_expected_behavior(matcher):
    pattern = [{"LEMMA": {"IN": instead_terms}}]
    matcher.add("S_EB_INSTEAD_OF_EXPECTED_BEHAVIOR", on_match_s_eb_instead_of_expected_behavior, pattern)


def on_match_s_eb_instead_of_expected_behavior(matcher, doc, i, matches):
    global count_s_eb_instead_of_expected_behavior
    count_s_eb_instead_of_expected_behavior = count_s_eb_instead_of_expected_behavior + 1


# S_EB_WOULD_BE
# Sentences describing desires with would and some adjective
# * (clause) it/this would/might (really) be [adjective] [predicate]
# * this would make [predicate]"
def setup_s_eb_would_be(matcher):
    pattern = [{"LEMMA": {"IN": would_terms}}, {"LOWER": "really", "OP": "?"}, {"LOWER": "be", "OP": "?"},
               {"POS": "ADJ", "OP": "?"}]
    matcher.add("S_EB_WOULD_BE", on_match_s_eb_would_be, pattern)


def on_match_s_eb_would_be(matcher, doc, i, matches):
    global count_s_eb_would_be
    count_s_eb_would_be = count_s_eb_would_be + 1


# S_EB_EXPECTED
# Sentence containing "expect" terms
# "(But,) [subject] expect(ed) that/to [predicate]
# [noun phrase] (verb to be) expecting [predicate]
# [(clauses)] [subject] would expect [predicate]
# The/My expectation [verb to be] [clause]"
def setup_s_eb_expected(matcher):
    pattern = [{"LEMMA": {"IN": expect_terms}}]
    matcher.add("S_EB_EXPECTED", on_match_s_eb_expected, pattern)


def on_match_s_eb_expected(matcher, doc, i, matches):
    global count_s_eb_expected
    count_s_eb_expected = count_s_eb_expected + 1


if __name__ == '__main__':

    issues = pd.read_csv('../data/bug_reports/github_data_2009.csv')

    print("Total issues:", len(issues["post"]))

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab, validate=True)

    setup_s_eb_should(matcher)
    setup_s_eb_exp_behavior(matcher)
    setup_s_eb_instead_of_expected_behavior(matcher)
    setup_s_eb_would_be(matcher)
    setup_s_eb_expected(matcher)

    issue_matches = []
    eb_list = []

    for index, issue in issues.iterrows():
        eb_str = ""
        match = False
        for sentence in nlp(issue["post"]).sents:
            sent = sentence.text.strip()
            if sent.startswith(">") or sent.endswith("?"):
                continue
            else:
                doc = nlp(sent)
                matches = matcher(doc)
                if len(matches) >= 1:
                    match = True
                    eb_str = eb_str + sent + " "

        if match:
            issue_matches.append(True)
            eb_list.append(eb_str)
        else:
            issue_matches.append(False)

    assert (len(issue_matches) == len(issues["post"]))
    print("# of S_EB_SHOULD sentences: ", count_s_eb_should)
    print("# of S_EB_EXP_BEHAVIOR sentences: ", count_s_eb_exp_behavior)
    print("# of S_EB_INSTEAD_OF_EXPECTED_BEHAVIOR sentences: ", count_s_eb_instead_of_expected_behavior)
    print("# of S_EB_WOULD_BE sentences: ", count_s_eb_would_be)
    print("# of S_EB_EXPECTED sentences: ", count_s_eb_expected)

    eb_issues = issues[issue_matches]
    assert (len(eb_issues) == len(eb_list))
    eb_issues = eb_issues.assign(EB=eb_list)
    eb_issues.to_csv("../data/bug_reports/github_data_EB.csv", index=False)
