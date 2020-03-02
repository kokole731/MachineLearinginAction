import numpy as np


def loadDataSet():
    """
    生成数据集
    :return: 句子和标签
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_voc_list(posting_list):
    """
    根据数据集生成词汇表
    :param posting_list: 数据集
    :return: 词汇表
    """
    voc_set = set()
    # 迭代生成词汇表
    for posting in posting_list:
        voc_set = voc_set | set(posting)
    return list(voc_set)


def word2vector(voc_list, word_list):
    """
    将句子转化为词向量
    :param voc_list: 词汇表
    :param word_list: 输入句子
    :return: 词向量
    """
    word_vector = [0] * len(voc_list)
    for word in word_list:
        if word in voc_list:
            index = voc_list.index(word)
            word_vector[index] = 1
        else:
            print('The word is not in the vocabulary list!')
    return word_vector


def get_train_mat(data_set):
    """
    获得训练矩阵
    :param data_set: 数据集
    :return: 训练矩阵
    """
    train_mat = []
    voc_list = create_voc_list(data_set)
    for posting in data_set:
        vector = word2vector(voc_list, posting)
        train_mat.append(vector)
    return train_mat


def trainNB(train_mat, labels):
    """
    训练数据集
    :param train_mat:训练矩阵(二维列表：row文档个数，col词汇表)
    :param labels: 文档标签（是否侮辱性）
    :return: 训练结果（各个单词在侮辱性/非侮辱性文档中出现的概率），侮辱性/非侮辱性文档占总文档的概率
    """
    # 训练集数据个数
    row = len(train_mat)
    # 单词向量长度
    col = len(train_mat[0])
    # 各个单词在非侮辱性文档中出现的次数,初始化为1
    word_num_vector_0 = np.ones(col)
    # 各个单词在侮辱性文档中出现的次数
    word_num_vector_1 = np.ones(col)
    # 非侮辱性和侮辱性文档中单词的总数
    word_sum_0 = 2
    word_sum_1 = 2
    # 遍历所有文档
    for i in range(row):
        # 该文档为非侮辱性文档
        if labels[i] == 0:
            # 该文档中的单词在非侮辱性文档单词出现的次数+1
            word_num_vector_0 += train_mat[i]
            # 非侮辱性文档单词总数加上这篇文档的单词总数
            word_sum_0 += sum(train_mat[i])
        if labels[i] == 1:
            word_num_vector_1 += train_mat[i]
            word_sum_1 += sum(train_mat[i])
    p_vector_0 = np.log(word_num_vector_0 / word_sum_0)
    p_vector_1 = np.log(word_num_vector_1 / word_sum_1)
    # 计算侮辱性和非侮辱性文档的出现概率
    p_1 = sum(labels) / len(labels)
    p_0 = 1 - p_1
    return p_vector_0, p_vector_1, p_0, p_1


def clf_nb(test_word_vector, p_vector_0, p_vector_1, p_0, p_1):
    """
    朴素贝叶斯分类器
    :param test_word_vector: 需要测试的句子
    :param p_vector_0: 各单词在非侮辱性文档中出现的概率（条件概率）
    :param p_vector_1: 各单词在侮辱性文档中出现的概率
    :param p_0: 非侮辱性文档的出现概率
    :param p_1: 侮辱性文档的出现概率
    :return:
    """
    p0 = sum(p_vector_0 * test_word_vector) + np.log(p_0)
    p1 = sum(p_vector_1 * test_word_vector) + np.log(p_1)
    if p0 > p1:
        return 0
    else:
        return 1


def word_test(test_word):
    """
    测试句子
    :param test_word: 输入句子
    :return: 是否为侮辱性文字
    """
    data_set, class_vector = loadDataSet()
    voc_list = create_voc_list(data_set)
    train_mat = get_train_mat(data_set)
    p_vector_0, p_vector_1, p_0, p_1 = trainNB(train_mat, class_vector)
    test_word_vector = np.array(word2vector(voc_list, test_word))
    rst = clf_nb(test_word_vector, p_vector_0, p_vector_1, p_0, p_1)
    if rst == 1:
        print(test_word, '侮辱性文字')
    else:
        print(test_word, '非侮辱性文字')
    return rst


if __name__ == '__main__':
    # posting_list, class_vector = loadDataSet()
    # voc_list = create_voc_list(posting_list)
    # test_posting_list = ['dog', 'has']
    # word_vector = word2vector(voc_list, test_posting_list)
    # train_mat = get_train_mat(posting_list)
    # trainNB(train_mat, class_vector)
    test1 = ['stupid', 'garbage']
    test2 = ['my', 'dog']
    word_test(test1)
    word_test(test2)
