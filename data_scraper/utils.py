import os
import sys


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


def question_identifier(sentence):
    start_words = ['who', 'what', 'when', 'where', 'why', 'which', 'how', "while", "do", "does", "did", "will", "would",
                   "can", "could", "shall", "should", "may", "might", "must", "is", "are", "were", "was", "has", "have",
                   "had"]
    flag = False
    for word in start_words:
        if sentence.lower().startswith(word.lower()):
            flag = True
            # return True
    if flag and sentence.endswith('?'):
        # print(sentence)
        return True
    return False
