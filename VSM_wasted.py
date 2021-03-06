import datetime
import json
import math
import nltk
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer


# 获取Lemmatization需要的词性
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None


def lemmatize_doc(doc):
    res = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for word, pos in nltk.pos_tag(doc):
        wordnet_pos = get_wordnet_pos(pos) or nltk.corpus.wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res


def tokenizer(doc):
    en_stopwords = nltk.corpus.stopwords.words('english')
    en_stopwords.extend([',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'])
    doc = nltk.word_tokenize(doc.lower())               # 分词
    doc = [w for w in doc if w not in en_stopwords]     # 去除停用词和标点
    doc = lemmatize_doc(doc)                            # 词形还原lemmatization
    return doc


def get_data(isModi):
    # 返回docs[列表]：内部每个元素都是list，储存分词后的每篇文章的内容
    # 返回queries[列表]：内部每个元素都是list，储存每个query的内容
    # 返回labels[ndarray数组]：内部每个元素都是ndarray数组，储存每个query对所有文档的相关性
    docs, queries, labels = [], [], []
    docs_id = {}
    num = 0

    if isModi == False:
        f_d = json.load(open("data/documents.json", 'r', encoding="utf-8", errors="ignore"))
        for key in f_d:
            docs.append(tokenizer(f_d[key]))
            docs_id[key] = num      # num是编号，key是json.load()转换成字典之后的key
            num += 1
        num_doc = num
        json_f = dict(zip(docs_id.keys(), docs))
        json_f = json.dumps(json_f)
        with open("data/modi_documents.json", 'w') as json_file:
            json_file.write(json_f)
    else:
        f_d = json.load(open("data/modi_documents.json", 'r', encoding="utf-8", errors="ignore"))
        for key in f_d:
            docs.append(f_d[key])
            docs_id[key] = num
            num += 1
        num_doc = num
    print("Length of docs: ", len(docs))
    print("docs[0]:", docs[0])
    print("type(docs[0]):", type(docs[0]))

    f = json.load(open("data/testset_no_label.json", 'r'))
    print("test set")

    for key in f["queries"]:
        queries.append(tokenizer(f["queries"][key]))
        label = f["labels"][key]
        label_ = np.zeros(num_doc)  # 获取一个num_doc大的数组，存储对于当前query，该文章是否是检索结果，初始化为0
        for i in range(len(label)):     # 一个query可能有多个label（答案）
            # label[i] = docs_id[str(label[i])]
            label_[docs_id[str(label[i])]] = 1  # 将对应文章id设为1，即该文章是当前query的结果
        # one_hot = np.eye(num_doc)[label]  # [x, num_doc]
        # label = np.sum(one_hot, axis=0)  # [num_doc]
        labels.append(label_)   # labels是一个二维数组，内部每个数组是对于每个query，通过01判断该文章是否是结果
    labels = np.array(labels)   # 转换为numpy的数组ndarray对象
    print("num_query:", len(queries))
    print("queries[0]:", queries[0])
    print("type(queries[0]):", type(queries[0]))

    return docs, queries, labels


class VSM(object):
    def __init__(self, docs):
        self.D = len(docs)
        # self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D   # 平均每个doc有多少单词
        self.docs = docs
        self.f = []         # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}        # 存储每个词及出现了该词的文档数量
        self.idf = {}       # 存储每个词的idf值
        self.tfidf = []     # 列表中每个元素是dict，代表每篇文章。每个dict存储该文章的每个词和它对应的tf-idf值（只存前10个作关键字）
        self.init()

    def init(self):
        """
        # 初始化部分，将corpus全部计算并写入文件中，加快速度。
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp:
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D)-math.log(v+1)
        json_f = json.dumps(self.idf)
        with open("data/doc_idf.json", 'w') as json_file:   # 保存docs的idf
            json_file.write(json_f)
        for index in range(self.D):
            tmp = {}
            for w in self.f[index]:
                tmp[w] = self.f[index][w] * self.idf[w]
            tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:10]
            self.tfidf.append(dict(tmp))
        json_f = json.dumps(self.tfidf)
        with open("data/doc_tfidf_10.json", 'w') as json_file:  # 保存表示docs的向量（取前10维）
            json_file.write(json_f)
        """
        # 初始化结束后可以只使用这部分代码，从文档中直接获得idf和docs的文档向量
        self.idf = json.load(open("data/doc_idf.json", 'r', encoding="utf-8", errors="ignore"))
        self.tfidf = json.load(open("data/doc_tfidf_10.json", 'r', encoding="utf-8", errors="ignore"))

    def cosine_sim(self, doc_tfidf, index):
        doc_vec = list(self.tfidf[index].values())
        doc_words = list(self.tfidf[index].keys())
        query_vec = []
        for w in doc_words:
            if w in doc_tfidf:
                query_vec.append(doc_tfidf[w])
            else:
                query_vec.append(0)
        dot_value = 0
        qv_norm = 0
        dv_norm = 0
        for i in range(len(doc_vec)):
            dot_value += query_vec[i] * doc_vec[i]
            qv_norm += query_vec[i] ** 2
            dv_norm += doc_vec[i] ** 2
        qv_norm = qv_norm ** 0.5
        dv_norm = dv_norm ** 0.5
        if qv_norm * dv_norm == 0:
            score = 0
        else:
            score = dot_value / (qv_norm * dv_norm)
        return score

    def simall(self, doc):
        scores = []
        doc_tf = {}
        doc_tfidf = {}
        for word in doc:
            if word in self.idf:
                doc_tf[word] = doc_tf.get(word, 0) + 1  # 存储该query中每个词的tf
        for word in doc_tf:
            doc_tfidf[word] = doc_tf[word] * self.idf[word]
        for index in range(self.D):
            score = self.cosine_sim(doc_tfidf, index)
            scores.append(score)
        return scores


def NDCG(logits, target, k):
    """
    Compute normalized discounted cumulative gain.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

    indices = np.argsort(-logits, 1)
    NDCG = 0
    for i in range(indices.shape[0]):
        DCG_ref = 0
        num_rel_docs = np.count_nonzero(target[i])
        for j in range(indices.shape[1]):
            if j == k:
                break
            if target[i, indices[i, j]] == 1:
                DCG_ref += 1 / np.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += 1 / np.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return NDCG / indices.shape[0]


def MRR(logits, target, k):
    """
    Compute mean reciprocal rank.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape

    # num_doc = logits.shape[1]
    indices_k = np.argsort(-logits, 1)[:, :k]  # 取topK 的index   [n, k]

    reciprocal_rank = 0
    for i in range(indices_k.shape[0]):
        for j in range(indices_k.shape[1]):
            if target[i, indices_k[i, j]] == 1:
                reciprocal_rank += 1.0 / (j + 1)
                break

    return reciprocal_rank / indices_k.shape[0]


if __name__ == '__main__':
    tic = datetime.datetime.now()
    docs, queries, labels = get_data(True)
    toc = datetime.datetime.now()
    print("data preprocess finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    s = VSM(docs)
    toc = datetime.datetime.now()
    print("VSM Model finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    scores = []
    labels = labels
    for query in queries:
        score = s.simall(query)
        scores.append(score)
        if len(scores) % 100 == 0:
            print(len(scores))
    logits = np.array(scores)

    # indices = np.argsort(-logits, 1)[:, :10]
    # np.save("2017xxx.npy", indices)    # 最终提交文件2017xxx.npy（学号命名）
    toc = datetime.datetime.now()
    print("logits finished in {}".format(toc - tic))

    tic = datetime.datetime.now()
    ndcg_10 = NDCG(logits, labels, 10)
    print('NDCG@10 - ', ndcg_10)

    mrr = MRR(logits, labels, 10)
    print('MRR@10 - ', mrr)

    # precision, recall, F1_Score = Precision_Recall_F1Score(logits, labels, 10)
    # print("precision@10 -  ", precision, "\trecall@10 - ", recall, "\tF1_Score@10 - ", F1_Score)
    toc = datetime.datetime.now()
    print("test finished in {}".format(toc - tic))
