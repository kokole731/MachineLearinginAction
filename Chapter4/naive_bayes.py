"""
实现朴素贝叶斯算法的简单实例
"""
import numpy as np


def create_data():
    """
    挂科不挂科案例
    :return: 数据
    """
    feature = np.array([
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 1]
    ])
    label = ['挂科', '不挂科', '不挂科', '挂科', '挂科', '不挂科', '不挂科', '挂科']
    return feature, label


def cal_p(train_feature, train_label, test_data):
    """
    计算测试样本标签为X的概率
    :param train_feature: 训练样本特征矩阵
    :param train_label: 训练样本标签列表
    :param test_data: 测试数据特征列表
    :return: 概率值
    """
    label_set = set(train_label)
    p_dict = {}
    # label为标签集合中的某个值
    for label in label_set:
        # 训练集的数据个数，特征个数
        row, col = train_feature.shape
        # 标签值为当前循环中标签值的所有样本索引
        test_label_feature_index = [example == label for example in train_label]
        # 为当前标签的所有样本（方便计算条件概率）
        y_sample = train_feature[test_label_feature_index]
        # y的概率（test_label的概率）
        p_y = y_sample.shape[0] / row
        product = 1
        for i in range(col):
            test_feature_i = test_data[i]
            num_list = y_sample[:, i] == test_feature_i
            # 条件概率
            p_i = list(num_list).count(True) / row
            p_i_y = p_i / p_y
            product *= p_i_y
        p_dict[label] = p_y * product
    sorted_dict_p = sorted(p_dict.items(), key=lambda item: item[1], reverse=True)
    return sorted_dict_p[0][0]


if __name__ == '__main__':
    feature, label = create_data()
    test = [0, 0, 1]
    p = cal_p(feature, label, test)
    print(p)
