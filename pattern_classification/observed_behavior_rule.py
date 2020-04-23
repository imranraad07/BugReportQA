import spacy
import pandas as pd
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
contrast_terms = ["although", "but", "however", "nevertheless", "though", "yet"]
conditional_terms = ["if", "when", "while", "whenever", "whereas", "upon"]
eb_modal_terms = ["must", "need", "should"]

count_s_ob_neg_aux_verb = 0
count_s_ob_verb_error = 0
count_s_ob_neg_verb = 0
count_s_ob_but = 0
count_s_ob_cond_pos = 0


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


# S_OB_BUT & S_OB_BUT_NEG
# Sentence with contrasting terms + affirmative clause
# Sentence with contrasting terms + negative predicates
# ([sentence],) [contrast term] [affirmative clause]
# Note: using the existence of a verb as an indication of a clause (affirmative or negative)
def setup_s_ob_but(matcher):
    but_pattern = [{"LOWER": {"IN": contrast_terms}},
                   {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
                   {"POS": "VERB"}]
    matcher.add("S_OB_BUT", on_match_s_ob_but, but_pattern)

def on_match_s_ob_but(matcher, doc, id, matches):
    global count_s_ob_but
    count_s_ob_but = count_s_ob_but + 1
    print('S_OB_BUT Matched!', doc.text)


# S_OB_COND_POS
# Compound sentence with conditional clause followed by a positive clause
# ([clause]) [conditional clause](,) [non-EB-modal POSITIVE sentence] ([clause])
# Note: using the existence of a verb as an indication of a clause (affirmative or negative)
def setup_s_ob_cond_pos(matcher):
    cond_pos_pattern = [{"LEMMA": {"IN": conditional_terms}},
                        {"LEMMA": {"NOT_IN": eb_modal_terms},  "OP": "+"},
                        {"POS": "VERB"}]
    matcher.add("S_OB_COND_POS", on_match_s_ob_cond_pos, cond_pos_pattern)

def on_match_s_ob_cond_pos(matcher, doc, id, matches):
    global count_s_ob_cond_pos
    count_s_ob_cond_pos = count_s_ob_cond_pos + 1
    print('S_OB_COND_POS Matched!', doc.text)



if __name__ == '__main__':

    issues = pd.read_csv('../data/bug_reports/github_data.csv',
                         header=None,
                         names = ['repo','issue_link','issue_id','post','question','answer'])

    print("Total issues:", len(issues["post"]))

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab, validate=True)
    setup_s_ob_neg_aux_verb(matcher)
    setup_s_ob_verb_error(matcher)
    setup_s_ob_neg_verb(matcher)
    setup_s_ob_but(matcher)
    setup_s_ob_cond_pos(matcher)

    issue_matches = []
    ob_list = []
    for index, issue in issues.iterrows():
        ob_str = ""
        sent_matches = False
        for sentence in nlp(issue["post"]).sents:
            sent = sentence.text.strip()
            if sent.startswith(">") or sent.endswith("?"):
                continue
            else:
                sent_nlp = nlp(sent)
                matches = matcher(sent_nlp)
                if (len(matches) >= 1):
                    sent_matches = True
                    ob_str = ob_str + sent

        if sent_matches:
            issue_matches.append(True)
            ob_list.append(ob_str)
        else:
            issue_matches.append(False)

    assert (len(issue_matches) == len(issues["post"]))
    print("All Patterns Percentage Matched: ", sum(issue_matches) * 100 / len(issues))
    print("S_OB_NEG_AUX_VERB sentences: ", count_s_ob_neg_aux_verb)
    print("S_OB_VERB_ERROR sentences: ", count_s_ob_verb_error)
    print("S_OB_NEG_VERB sentences: ", count_s_ob_neg_verb)
    print("S_OB_BUT sentences: ", count_s_ob_but)
    print("S_OB_COND_POS sentences: ", count_s_ob_cond_pos)

    ob_issues = issues[issue_matches]
    ob_issues.assign(post=ob_list)
    ob_issues.to_csv("output.csv", index=False)