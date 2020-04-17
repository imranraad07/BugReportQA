import csv
import sys
import spacy

negative_aux_verbs = ["are not", "aren't", "ain't", "can not", "cannot", "can't", "could not", "couldn't", "does not",
                      "doesn't", "did not", "didn't", "has not", "hasn't", "had not", "hadn't", "have not", "haven't",
                      "is not", "isn't", "was not", "wasn't", "were not", "weren't", "will not", "willn't", "won't"]

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


def check_exits(sentence, check_list):
    return any(x in sentence for x in check_list)


def is_question(sentence):
    if sentence.endswith('?'):
        return True
    return False


# S_OB_NEG_AUX_VERB
# Negative (simple) sentence with auxiliary verbs
# ([subject]) [negative auxiliary verb] [verb] ([complement])
def check_s_ob_neg_aux_verb(sentence):
    if len(sentence.text) > 300:
        return False
    elif is_question(sentence.text):
        return False
    sen_structure_set1 = ['AUX', 'PART', 'VERB']
    sen_structure_set2 = ['AUX', 'PART', 'ADJ', 'VERB']
    sen_structure_set3 = ['AUX', 'PART', 'ADV', 'VERB']
    if check_exits(sentence.text, negative_aux_verbs):
        postag = []
        for token in sentence:
            postag.append(token.pos_)
        if set(sen_structure_set1).issubset(postag) or set(sen_structure_set2).issubset(postag) or set(
                sen_structure_set3).issubset(postag):
            return True
    return False


# S_OB_NEG_VERB
# (Compound) sentence with a non-auxiliary negative verb
# (pre-clause) (subject/noun phrase) ([adjective/adverb]) [negative verb] ([complement])
def check_s_ob_neg_verb(sentence):
    if len(sentence.text) > 300:
        return False
    elif is_question(sentence.text):
        return False
    sen_structure_set1 = ['ADJ', 'VERB']
    sen_structure_set2 = ['ADV', 'VERB']
    if check_exits(sentence.text, negative_verbs):
        postag = []
        for token in sentence:
            postag.append(token.pos_)
        if set(sen_structure_set1).issubset(postag) or set(sen_structure_set2).issubset(postag):
            return True
    return False


# S_OB_VERB_ERROR
# (Compound) sentence with verb phrase using error and no [negative auxiliary verb]
# (clause) ([subject]) [verb] ([personal pronoun]) ([prep]) [ERROR_NOUN_PHRASE] [predicate]
def check_s_ob_verb_error(sentence):
    if len(sentence.text) > 300:
        return False

    if is_question(sentence.text):
        return False

    if check_exits(sentence.text, error_terms):
        sen_structure_set1 = ['VERB']
        for i,token in enumerate(sentence):
            if token.text in error_terms:
                predicates = sentence[i+1:]
                # print(predicates)
                postag = []
                for predicate in predicates:
                    postag.append(predicate.pos_)
                # need to do a few tweaks here
                if set(sen_structure_set1).issubset(postag):
                    # print(token)
                    # print(sentence)
                    # print("------------")
                    # print(predicates[1])
                    # print(postag)
                    return True
    return False


if __name__ == '__main__':
    issues = []
    issue_links = []

    csv.field_size_limit(sys.maxsize)
    with open('../data/results/github_data.csv') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        for row in csvReader:
            if not row:
                continue
            if row[1] in issue_links:
                continue
            issue_links.append(row[1])
            issues.append(row[2])

    print("Total issues:", len(issue_links))

    nlp = spacy.load("en_core_web_sm")
    count_s_ob_neg_aux_verb_rule_1 = 0
    count_s_ob_neg_verb_rule_2 = 0
    count_s_ob_verb_error_rule_3 = 0
    count_rule_1_rule_2 = 0
    count_rule_1_rule_3 = 0
    count_rule_2_rule_3 = 0
    no_rule = 0
    for issue in issues:
        in_rule_1 = False
        in_rule_2 = False
        in_rule_3 = False
        in_rule_1_rule_2 = False
        in_rule_1_rule_3 = False
        in_rule_2_rule_3 = False
        in_no_rule = False
        for sentence in nlp(issue).sents:
            if sentence.text.startswith(">"):
                continue
            else:
                flag_1 = False
                flag_2 = False
                flag_3 = False
                # S_OB_NEG_AUX_VERB
                if check_s_ob_neg_aux_verb(sentence):
                    flag_1 = True
                    if not in_rule_1:
                        count_s_ob_neg_aux_verb_rule_1 = count_s_ob_neg_aux_verb_rule_1 + 1
                        in_rule_1 = True

                # S_OB_NEG_VERB
                if check_s_ob_neg_verb(sentence):
                    flag_2 = True
                    if not in_rule_2:
                        count_s_ob_neg_verb_rule_2 = count_s_ob_neg_verb_rule_2 + 1
                        in_rule_2 = True

                # S_OB_VERB_ERROR
                if check_s_ob_verb_error(sentence):
                    flag_3 = True
                    if not in_rule_3:
                        count_s_ob_verb_error_rule_3 = count_s_ob_verb_error_rule_3 + 1
                        in_rule_3 = True

                if flag_1 and flag_2 and not in_rule_1_rule_2:
                    count_rule_1_rule_2 = count_rule_1_rule_2 + 1
                    in_rule_1_rule_2 = True
                if flag_1 and flag_3 and not in_rule_1_rule_3:
                    count_rule_1_rule_3 = count_rule_1_rule_3 + 1
                    in_rule_1_rule_3 = True
                if flag_2 and flag_3 and not in_rule_2_rule_3:
                    count_rule_2_rule_3 = count_rule_2_rule_3 + 1
                    in_rule_2_rule_3 = True

                if flag_1 or flag_2 or flag_3:
                    in_no_rule = True
        if not in_no_rule:
            no_rule = no_rule + 1

    print("rule 1: S_OB_NEG_AUX_VERB", count_s_ob_neg_aux_verb_rule_1)
    print("rule 2: S_OB_NEG_VERB", count_s_ob_neg_verb_rule_2)
    print("rule 3: S_OB_VERB_ERROR", count_s_ob_verb_error_rule_3)
    print("rule S_OB_NEG_AUX_VERB and S_OB_NEG_VERB", count_rule_1_rule_2)
    print("rule S_OB_NEG_AUX_VERB and S_OB_VERB_ERROR", count_rule_1_rule_3)
    print("rule S_OB_NEG_VERB and S_OB_VERB_ERROR", count_rule_2_rule_3)
    print("rule not fit", no_rule)
    print("Total issues:", len(issues))
    print("percentage:", (len(issues) - no_rule) * 100 / len(issues))
