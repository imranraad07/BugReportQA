# https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
import csv
import sys

from gensim import corpora, models, similarities
import jieba
import re

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def filerSentence(sentence):
    porter = PorterStemmer()
    stopWords = set(stopwords.words('english'))
    res = ''.join([i for i in sentence if not i.isdigit()])
    res = re.sub(r'\W+', ' ', res)
    ret = ''
    texts = word_tokenize(res)
    for word in texts:
        word = word.strip()
        if word not in stopWords:
            if len(word) > 2:
                ret = ret + porter.stem(word.lower()) + " "
    # print(ret, " ", sentence)
    return ret


if __name__ == '__main__':
    issues = []
    comments = []

    originalIssues = []
    originalComments = []

    csv.field_size_limit(sys.maxsize)
    with open('results/github_data_sample.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            # if len(row[3]) > 300:
            #     continue

            issues.append(filerSentence(row[1]))
            comments.append(filerSentence(row[3]))
            originalIssues.append(row[1])
            originalComments.append(row[3])

    print(len(issues))
    # print(comments)

    texts = [jieba.lcut(text) for text in issues]
    dictionary = corpora.Dictionary(texts)
    print(dictionary)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

    result_folder = "results"
    result_file = "tditf_github_data.csv"
    comment_number = 0
    unique_comments = 0
    idx = 0
    for comment in comments:
        comment_number = comment_number + 1
        keyword = comment
        kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
        sim = index[tfidf[kw_vector]]
        unique = False
        for i in range(len(sim)):
            if sim[i] > 0.35:
                # print('comment {0} is similar to text{1}: {2}'.format(comment_number, i + 1, sim[i]))
                if i + 1 == comment_number:
                    unique = True
        if unique:
            unique_comments = unique_comments + 1
            sw = csv.writer(open('{0}/{1}'.format(result_folder, result_file), 'a'))
            sw.writerow([
                '{0}'.format(originalIssues[idx]),
                '{0}'.format(originalComments[idx]),
                '{0}'.format(sim[comment_number - 1])
            ])
        idx = idx + 1
    print(unique_comments)