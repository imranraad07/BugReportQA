# https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
import csv
import sys

from gensim import corpora, models, similarities
import jieba
import re

from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def filter_sentence(sentence):
    porter = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    java_keywords = ["abstract", "assert", "boolean",
                     "break", "byte", "case", "catch", "char", "class", "const",
                     "continue", "default", "do", "double", "else", "extends", "false",
                     "final", "finally", "float", "for", "goto", "if", "implements",
                     "import", "instanceof", "int", "interface", "long", "native",
                     "new", "null", "package", "private", "protected", "public",
                     "return", "short", "static", "strictfp", "super", "switch",
                     "synchronized", "this", "throw", "throws", "transient", "true",
                     "try", "void", "volatile", "while"]
    res = ''.join([i for i in sentence if not i.isdigit()])
    res = re.sub(r'\W+', ' ', res)
    ret = ''
    texts = word_tokenize(res)
    for word in texts:
        word = word.strip()
        if word not in stopWords and word not in java_keywords:
            if len(word) > 2:
                ret = ret + porter.stem(word.lower()) + " "
    return ret


def modify_comment(text):
    modified_text = text.replace("\n", " ")
    modified_text = re.sub(r'```.+```', '', modified_text)
    return modified_text


if __name__ == '__main__':
    issues = []
    questions = []
    answers = []
    issue_links = []

    originalIssues = []
    originalQuestions = []
    originalAnswers = []

    csv.field_size_limit(sys.maxsize)
    with open('results/github_data.csv') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        for row in csvReader:
            if row[1] in issue_links:
                continue
            issues.append(filter_sentence(row[2]))
            questions.append(filter_sentence(modify_comment(row[3])))
            answers.append(filter_sentence(modify_comment(row[4])))
            originalIssues.append(row[2])
            originalQuestions.append(row[3])
            originalAnswers.append(row[4])
            issue_links.append(row[1])

    print(len(issues))

    # space removal from texts
    texts_issues = [jieba.lcut(text) for text in issues]
    for text in texts_issues:
        while " " in text:
            text.remove(" ")
    dictionary_issue = corpora.Dictionary(texts_issues)
    print(dictionary_issue)
    feature_cnt_issue = len(dictionary_issue.token2id)
    corpus_issue = [dictionary_issue.doc2bow(text) for text in texts_issues]
    tfidf_issue = models.TfidfModel(corpus=corpus_issue)
    index_issue = similarities.SparseMatrixSimilarity(tfidf_issue[corpus_issue], num_features=feature_cnt_issue)

    # space removal from texts
    texts_questions = [jieba.lcut(text) for text in questions]
    for text in texts_questions:
        while " " in text:
            text.remove(" ")
    dictionary_questions = corpora.Dictionary(texts_questions)
    print(dictionary_questions)
    feature_cnt_questions = len(dictionary_questions.token2id)
    corpus_questions = [dictionary_questions.doc2bow(text) for text in texts_questions]
    tfidf_questions = models.TfidfModel(corpus=corpus_questions)
    index_questions = similarities.SparseMatrixSimilarity(tfidf_questions[corpus_questions],
                                                          num_features=feature_cnt_questions)

    result_folder = "results"
    result_file = "tditf_github_data.csv"
    comment_number = 0
    unique_comments = 0
    idx = 0
    for question in questions:
        kw_vector_issue = dictionary_issue.doc2bow(jieba.lcut(question))
        sim_issue = index_issue[tfidf_issue[kw_vector_issue]]
        unique = False
        if sim_issue[idx] > 0.10:
            unique_comments = unique_comments + 1

            kw_vector_question = dictionary_questions.doc2bow(jieba.lcut(answers[idx]))
            sim_question = index_questions[tfidf_questions[kw_vector_question]]

            sw = csv.writer(open('{0}/{1}'.format(result_folder, result_file), 'a'))
            sw.writerow([
                '{0}'.format(issue_links[idx]),
                '{0}'.format(originalIssues[idx]),
                '{0}'.format(originalQuestions[idx]),
                '{0}'.format(sim_issue[idx]),
                '{0}'.format(originalAnswers[idx]),
                '{0}'.format(sim_question[idx]),
            ])
        idx = idx + 1
    print(unique_comments)
