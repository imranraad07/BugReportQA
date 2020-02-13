# https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
import csv

from gensim import corpora, models, similarities
import jieba

if __name__ == '__main__':

    issues = []
    comments = []

    with open('results/github_data_sample.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            issues.append(row[1])
            comments.append(row[3])

    # print(issues)
    # print(comments)

    texts = [jieba.lcut(text) for text in issues]
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

    comment_number = 0
    unique_comments = 0
    for comment in comments:
        comment_number = comment_number + 1
        keyword = comment
        kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
        sim = index[tfidf[kw_vector]]
        unique = False
        for i in range(len(sim)):
            if sim[i] > 0.70:
                print('comment {0} is similar to text{1}: {2}'.format(comment_number, i + 1, sim[i]))
                # if i + 1 == comment_number:
                unique = True
        if unique:
            unique_comments = unique_comments + 1
    print(unique_comments)

# 0.2 threshold: same issue: 169
# 0.3 threshold: same issue: 79
# 0.4 threshold: same issue: 53
# 0.5 threshold: same issue: 40
# 0.6 threshold: same issue: 30
# 0.7 threshold: same issue: 22
#
# 0.2 threshold: different issue: 409
# 0.3 threshold: different issue: 156
# 0.4 threshold: different issue: 82
# 0.5 threshold: different issue: 58
# 0.6 threshold: different issue: 40
# 0.7 threshold: different issue: 26