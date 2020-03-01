import numpy as np
import operator
import os
import matplotlib.pyplot as plt


# 简单测试
def create_data_set():
    """

    :rtype: 样本特征、标签
    """
    group = np.array([[1, 101],
                      [5, 89],
                      [108, 5],
                      [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


# knn算法
def knn2(inx, data_set, labels, k):
    """
    knn algorithm
    :param inx: test data
    :param data_set: feature matrix
    :param labels: training data
    :param k: k-objects
    :return: best fit example's label
    """
    size = data_set.shape[0]
    input_mat = np.tile(inx, (size, 1)) - data_set
    sqr_dist_mat = input_mat ** 2
    dist_mat = sqr_dist_mat.sum(axis=1) ** 0.5
    # 给出从小到大排列的索引值
    sort_indices = dist_mat.argsort()
    class_count = {}
    for i in range(k):
        label = labels[sort_indices[i]]
        # 字典中该键的值+=1
        class_count[label] = class_count.get(label, 0) + 1
    # sorted_class: 按计数大小排列的字典 eg:('romantic', 2)
    sorted_class = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_class[0][0]


def knn(inx, data_set, labels, k):
    """
    :rtype: best fit label
    :param inx: 输入样本特征
    :param data_set: 训练数据集
    :param labels: 训练样本标签
    :param k: 最近邻居的数目
    """
    # 训练样本的个数
    data_set_size = data_set.shape[0]
    # 将输入数据行方向上复制四行
    diff_mat = np.tile(inx, (data_set_size, 1)) - data_set
    sqr_diff_mat = diff_mat ** 2
    sqr_diff_sum = sqr_diff_mat.sum(axis=1)
    distance_mat = sqr_diff_sum ** 0.5
    sorted_dist_indices = distance_mat.argsort()
    class_count = {}
    for i in range(k):
        label = labels[sorted_dist_indices[i]]
        class_count[label] = class_count.get(label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 将约会数据文本文件数据转换为矩阵
def file_to_matrix(filename):
    """
    将文本文件中的数据转化为matrix
    :param filename: the name of the file
    """
    fr = open(filename)
    array_lines = fr.readlines()
    number_of_line = len(array_lines)
    features = np.zeros((number_of_line, 3), dtype=float)
    label_vector = []
    index = 0
    for line in array_lines:
        # 去除回车字符
        line = line.strip()
        list_from_line = line.split('\t')
        features[index, :] = list_from_line[:3]
        label_vector.append(list_from_line[-1])
        index += 1
    return features, label_vector


# 选取特征值对应标签值画出散点图
def plot_scatter(features, labels):
    """
    plot the data
    :param features: matrix
    :param labels: list
    """
    feature_1 = features[:, 1]
    feature_2 = features[:, 2]
    plt.scatter(feature_1, feature_2, s=15 * np.array(labels), c=np.array(labels))
    plt.show()


# 标准化数据，使各个特征值具有相同的权值
def normalize(features):
    """
    normalize the data to 0~1
    :param features: array
    :return: normalized features
    """
    size = features.shape[0]
    # 所有行里面找max,min
    min_matrix = features.min(axis=0)
    max_matrix = features.max(axis=0)
    range_matrix = max_matrix - min_matrix
    min_matrix = np.tile(min_matrix, (size, 1))
    range_matrix = np.tile(range_matrix, (size, 1))
    features = (features - min_matrix) / range_matrix
    return features


# 约会数据测试
def date_test(rate, k):
    """
    好感度测试
    :param rate: 测试集比率
    :param k: kNN参数值
    """
    test_rate = rate
    file_fir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(file_fir, 'datingTestSet.txt')
    features, labels = file_to_matrix(filename)
    features = normalize(features)
    size = features.shape[0]
    test_num = int(size * test_rate)
    test_features = features[:test_num, :]
    test_labels = labels[:test_num]
    train_features = features[test_num:, :]
    train_labels = labels[test_num:]
    error_count = 0
    for i in range(test_num):
        ans = knn2(test_features[i], train_features, train_labels, k)
        print("the test result is %s, true answer is %s" \
              % (ans, test_labels[i]))
        if ans != test_labels[i]:
            error_count += 1
    print("test num is %d, error num is %d, error rate is %d%%" \
          % (test_num, error_count, error_count / test_num * 100))


# 将数字文本文件数据转换为特征矩阵和标签
def img_to_vector(filename):
    label = int(filename.split('_')[0][-1])
    fr = open(filename)
    file_lines = fr.readlines()
    row_num = len(file_lines)
    col_num = len(file_lines[0]) - 1
    feature = np.zeros((1, row_num * col_num))
    index = 0
    for line in file_lines:
        line = line.strip()
        for i in range(len(line)):
            feature[0, index] = int(line[i])
            index += 1
    return feature, label


def create_data(dir_name):
    """
    根据文件创建数据集(1024维特征值)
    :param dir_name: 训练/测试数据集文件夹名称
    :return: 特征矩阵(row*num_of_features),标签(true number)
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(dirname, 'digits', dir_name)
    digits_files = os.listdir(data_dir)
    size = len(digits_files)
    data_mat = np.zeros((size, 1024))
    labels = []
    index = 0
    for filename in digits_files:
        filename = os.path.join(data_dir, filename)
        feature, label = img_to_vector(filename)
        data_mat[index, :] = feature
        labels.append(label)
        index += 1
    return data_mat, labels


# 识别数字程序
def recognize_digit():
    training_features, training_labels = create_data('trainingDigits')
    test_features, test_labels = create_data('testDigits')
    error_count = 0
    test_num = test_features.shape[0]
    for i in range(test_num):
        test_rst = knn2(test_features[i], training_features, training_labels, 3)
        print('test result is %d, right answer is %d'
              % (test_rst, test_labels[i]))
        if test_rst != test_labels[i]:
            error_count += 1
    print("识别数字%d个，识别正确%d个，识别错误%d个，正确率为%.5f%%"
          % (test_num, test_num - error_count, error_count, (1 - error_count / test_num) * 100))


if __name__ == '__main__':
    recognize_digit()
