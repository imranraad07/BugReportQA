# https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
# https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.Xk-JKnVKg5k
import csv
import re
import sys

import jieba
from gensim import corpora, models, similarities
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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

    ret = ''
    texts = word_tokenize(sentence)
    for word in texts:
        word = word.strip()
        if word not in stopWords and word not in java_keywords:
            stemmed_word = porter.stem(word)
            if len(stemmed_word) > 2:
                ret = ret + stemmed_word + " "
    return ret


def modify_comment(text):
    # remove new lines with space
    modified_text = text.replace("\n", " ")
    # remove codes
    modified_text = re.sub(r'```.+```', '', modified_text)
    # remove non-ascii characters
    modified_text = re.sub("([^\x00-\x7F])+", " ", modified_text)
    # lower case
    modified_text = modified_text.lower()
    # remove special characters
    modified_text = re.sub('[^A-Za-z]+', ' ', modified_text)
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
    with open('results/github_data_sample.csv') as csvDataFile:
        csvReader = csv.reader((line.replace('\0', '') for line in csvDataFile))
        for row in csvReader:
            if row[1] in issue_links:
                continue
            issues.append(filter_sentence(modify_comment(row[2])))
            questions.append(filter_sentence(modify_comment(row[3])))
            answers.append(filter_sentence(modify_comment(row[4])))
            originalIssues.append(row[2])
            originalQuestions.append(row[3])
            originalAnswers.append(row[4])
            issue_links.append(row[1])

    print(len(issues))

    texts_issues = [text.split() for text in issues]
    dictionary_issue = corpora.Dictionary(texts_issues)
    print(dictionary_issue)
    feature_cnt_issue = len(dictionary_issue.token2id)
    corpus_issue = [dictionary_issue.doc2bow(text) for text in texts_issues]
    tfidf_issue = models.TfidfModel(corpus=corpus_issue, normalize=True)
    index_issue = similarities.SparseMatrixSimilarity(tfidf_issue[corpus_issue], num_features=feature_cnt_issue)

    print(dictionary_issue.token2id)

    # print(summarize(' '.join(originalIssues), word_count=200))
    # print(len(keywords('. '.join(issues))))

    # model = Word2Vec(texts_issues, min_count=0, workers=cpu_count())
    # print(model)
    # print(model.most_similar(PorterStemmer().stem("mongodb"), topn=3))

    # pprint.pprint(dictionary_issue.token2id)

    # for doc in corpus_issue:
    #     print(doc)

    # d = {dictionary_issue.get(id): value for doc in tfidf_issue[corpus_issue] for id, value in doc}
    # d = sorted(d.items(), reverse=True, key=lambda x: x[1])
    # print(d)

    # model = gensim.models.Word2Vec(texts_issues, size=150, window=10, min_count=2, workers=10, iter=10)
    # print(model.wv.most_similar(positive=["master"], topn=5))
    # print(model.wv.similarity("json", PorterStemmer().stem("annotation")))

    # sample_text = issues[1] + "      " + questions[1]
    # tfidf_values = dict(tfidf_issue[dictionary_issue.doc2bow(word_tokenize(sample_text))])
    # print(sample_text)
    # pprint.pprint(tfidf_values)

    texts_questions = [text.split() for text in questions]
    dictionary_questions = corpora.Dictionary(texts_questions)
    print(dictionary_questions)
    feature_cnt_questions = len(dictionary_questions.token2id)
    corpus_questions = [dictionary_questions.doc2bow(text) for text in texts_questions]
    tfidf_questions = models.TfidfModel(corpus=corpus_questions, normalize=True)
    index_questions = similarities.SparseMatrixSimilarity(tfidf_questions[corpus_questions],
                                                          num_features=feature_cnt_questions)

    # d = {dictionary_questions.get(id): value for doc in tfidf_questions[corpus_questions] for id, value in doc}
    # d = sorted(d.items(), reverse=True, key=lambda x: x[1])
    # print(d)

    result_folder = "results"
    result_file = "tditf_github_data.csv"
    comment_number = 0
    unique_comments = 0
    idx = 0
    for question in questions:
        kw_vector_issue_question = dictionary_issue.doc2bow(jieba.lcut(question))
        sim_issue_question = index_issue[tfidf_issue[kw_vector_issue_question]]
        unique = False

        if sim_issue_question[idx] > 0.20:
            unique_comments = unique_comments + 1

            kw_vector_question_answer = dictionary_questions.doc2bow(jieba.lcut(answers[idx]))
            sim_question_answer = index_questions[tfidf_questions[kw_vector_question_answer]]

            sample_text = issues[idx] + "      " + questions[idx]
            tfidf_values = dict(tfidf_issue[dictionary_issue.doc2bow(word_tokenize(sample_text))])
            tfidf_values = sorted(tfidf_values.items(), reverse=True, key=lambda x: x[1])[:3]

            top_terms = []
            for value in tfidf_values:
                top_terms.append(dictionary_issue[value[0]])
            # print(top_terms)

            sw = csv.writer(open('{0}/{1}'.format(result_folder, result_file), 'a'))
            sw.writerow([
                '{0}'.format(issue_links[idx]),
                '{0}'.format(originalIssues[idx]),
                '{0}'.format(originalQuestions[idx]),
                '{0}'.format(sim_issue_question[idx]),
                '{0}'.format(originalAnswers[idx]),
                '{0}'.format(sim_question_answer[idx]),
                '{0}'.format(top_terms)
            ])

        idx = idx + 1
    print(unique_comments)
