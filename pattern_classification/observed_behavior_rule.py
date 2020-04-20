import csv
import sys
import spacy
from spacy.matcher import Matcher

# negative_aux_verbs = ["are not", "aren't", "ain't", "can not", "cannot", "can't", "could not", "couldn't", "does not",
#                       "doesn't", "did not", "didn't", "has not", "hasn't", "had not", "hadn't", "have not", "haven't",
#                       "is not", "isn't", "was not", "wasn't", "were not", "weren't", "will not", "willn't", "won't"]

negative_verbs = ["affect", "break", "block", "bypass", "clear", "clobber", "close", "complain", "consume", "crash",
                  "cut", "delete", "delay", "die", "disappear", "enforce", "erase", "exit", "expire", "fail", "flicker",
                  "freeze", "forget", "glitch", "grow", "hang", "ignore", "increase", "interfere", "lack of", "lose",
                  "jerk", "jitter", "mess up", "mishandle", "mute", "offset", "overlap", "pause", "prohibit", "reduce",
                  "refuse", "remain", "reject", "rest", "restart", "revert", "skip", "stop", "stuck up", "suffer",
                  "throw", "time out", "truncate", "vanished", "wipe out", "terminate", "trim"]

error_terms = ["ambiguity", "breakage", "bug", "collision", "conflict", "confusion", "crash", "disaster",
               "error", "exception", "failure", "fault", "frustration", "glitch", "inability", "issue", "leak",
               "leakage", "lock", "loss", "mistake", "NPE", "null", "omission", "pain", "peek", "problem", "race",
               "rarity", "runaway", "segfault", "segmentation", "spam", "status", "symptom", "truncation", "typo",
               "violation", "wait", "warning", "zombie"]

count_s_ob_neg_aux_verb = 0
count_s_ob_verb_error = 0
count_s_ob_neg_verb = 0

# def check_exits(sentence, check_list):
#     return any(x in sentence for x in check_list)
#
# def is_question(sentence):
#     if sentence.endswith('?'):
#         return True
#     return False


# S_OB_NEG_AUX_VERB
# Negative (simple) sentence with auxiliary verbs
# ([subject]) [negative auxiliary verb] [verb] ([complement])
def setup_s_ob_neg_aux_verb(matcher):
    aux_verb_pattern = [{"LEMMA": {"IN": ["is","do","have","can"]}},
                        {"POS": "PART", "TEXT": "not"},
                        {"POS": "VERB"}]
    matcher.add("S_OB_NEG_AUX_VERB", on_match_s_ob_neg_aux_verb, aux_verb_pattern)

def on_match_s_ob_neg_aux_verb(matcher, doc, id, matches):
    global count_s_ob_neg_aux_verb
    count_s_ob_neg_aux_verb = count_s_ob_neg_aux_verb + 1
    print('S_OB_NEG_AUX_VERB Matched!', doc.text)

# def check_s_ob_neg_aux_verb(sentence):
#     if len(sentence.text) > 300:
#         return False
#     elif is_question(sentence.text):
#         return False
#     sen_structure_set1 = ['AUX', 'PART', 'VERB']
#     sen_structure_set2 = ['AUX', 'PART', 'ADJ', 'VERB']
#     sen_structure_set3 = ['AUX', 'PART', 'ADV', 'VERB']
#     if check_exits(sentence.text, negative_aux_verbs):
#         postag = []
#         for token in sentence:
#             postag.append(token.pos_)
#         if set(sen_structure_set1).issubset(postag) or set(sen_structure_set2).issubset(postag) or set(
#                 sen_structure_set3).issubset(postag):
#             return True
#     return False


# S_OB_VERB_ERROR
# (Compound) sentence with verb phrase using error and no [negative auxiliary verb]
# (clause) ([subject]) [verb] ([personal pronoun]) ([prep]) [ERROR_NOUN_PHRASE] [predicate]
def setup_s_ob_verb_error(matcher):
    verb_error_pattern = [{"POS": "VERB"},
                          {"POS": "PRON", "OP": "?"},
                          {"POS": "DET", "OP": "?"},
                          {"LEMMA": {"IN": error_terms}}]
    matcher.add("S_OB_VERB_ERROR", on_match_s_ob_verb_error, verb_error_pattern)

def on_match_s_ob_verb_error(matcher, doc, id, matches):
    global count_s_ob_verb_error
    count_s_ob_verb_error = count_s_ob_verb_error + 1
    print('S_OB_VERB_ERROR Matched!', doc.text)

# def check_s_ob_verb_error(sentence):
#     if len(sentence.text) > 300:
#         return False
#
#     if is_question(sentence.text):
#         return False
#
#     if check_exits(sentence.text, error_terms):
#         sen_structure_set1 = ['VERB']
#         for i,token in enumerate(sentence):
#             if token.text in error_terms:
#                 predicates = sentence[i+1:]
#                 # print(predicates)
#                 postag = []
#                 for predicate in predicates:
#                     postag.append(predicate.pos_)
#                 # need to do a few tweaks here
#                 if set(sen_structure_set1).issubset(postag):
#                     # print(token)
#                     # print(sentence)
#                     # print("------------")
#                     # print(predicates[1])
#                     # print(postag)
#                     return True
#     return False





# S_OB_NEG_VERB
# (Compound) sentence with a non-auxiliary negative verb
# (pre-clause) (subject/noun phrase) ([adjective/adverb]) [negative verb] ([complement])
def setup_s_ob_neg_verb(matcher):
    neg_verb_pattern = [{"POS": "VERB", "LEMMA": {"IN": negative_verbs}}]
    matcher.add("S_OB_NEG_VERB", on_match_s_ob_neg_verb, neg_verb_pattern)

def on_match_s_ob_neg_verb(matcher, doc, id, matches):
    global count_s_ob_neg_verb
    count_s_ob_neg_verb = count_s_ob_neg_verb + 1
    print('S_OB_NEG_VERB Matched!', doc.text)

# def check_s_ob_neg_verb(sentence):
#     if len(sentence.text) > 300:
#         return False
#     elif is_question(sentence.text):
#         return False
#     sen_structure_set1 = ['ADJ', 'VERB']
#     sen_structure_set2 = ['ADV', 'VERB']
#     if check_exits(sentence.text, negative_verbs):
#         postag = []
#         for token in sentence:
#             postag.append(token.pos_)
#         if set(sen_structure_set1).issubset(postag) or set(sen_structure_set2).issubset(postag):
#             return True
#     return False




def read_input():
    issues = []
    issue_links = []
    csv.field_size_limit(sys.maxsize)
    with open('../data/bug_reports/github_data.csv') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        for row in csvReader:
            if not row:
                continue
            if row[1] in issue_links:
                continue
            issue_links.append(row[1])
            issues.append(row[2])
    return issues, issue_links

if __name__ == '__main__':
    issues, issue_links = read_input()

    print("Total issues:", len(issue_links))

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab, validate=True)
    setup_s_ob_neg_aux_verb(matcher)
    setup_s_ob_verb_error(matcher)
    setup_s_ob_neg_verb(matcher)

    issue_matches = []
    for issue in issues:
        sent_matches = False
        for sentence in nlp(issue).sents:
            if sentence.text.startswith(">"):
                continue
            else:
                sent_nlp = nlp(sentence.text)
                matches = matcher(sent_nlp)
                if (len(matches) >= 1):
                    sent_matches = True

        if sent_matches:
            issue_matches.append(1)
        else:
            issue_matches.append(0)


    assert (len(issue_matches) == len(issues))
    print("All Patterns Percentage Matched: ", sum(issue_matches) * 100 / len(issues))
    print("S_OB_NEG_AUX_VERB sentences: ", count_s_ob_neg_aux_verb)
    print("S_OB_VERB_ERROR sentences: ", count_s_ob_verb_error)
    print("S_OB_NEG_VERB sentences: ", count_s_ob_neg_verb)