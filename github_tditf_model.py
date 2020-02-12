# https://medium.com/better-programming/introduction-to-gensim-calculating-text-similarity-9e8b55de342d
# https://www.machinelearningplus.com/nlp/gensim-tutorial/
from gensim import corpora, models, similarities
import jieba

if __name__ == '__main__':
    texts = ['I love reading Japanese novels. My favorite Japanese writer is Tanizaki Junichiro.',
             'Natsume Soseki is a well-known Japanese novelist and his Kokoro is a masterpiece.',
             'American modern poetry is good. ']
    keyword = 'Japan has some great novelists. Who is your favorite Japanese writer?'
    texts = [jieba.lcut(text) for text in texts]
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[kw_vector]]
    for i in range(len(sim)):
        print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))
