import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    """
    从文本文件中加载数据集
    :return: 列表型数据，标签
    """
    fr = open('testSet.txt')
    lines = fr.readlines()
    data_set = []
    label_list = []
    for line in lines:
        line = line.strip()
        element_list = line.split('\t')
        data_set.append([1.0, float(element_list[0]), float(element_list[1])])
        label_list.append(int(element_list[-1]))
    return data_set, label_list


def sigmoid(x):
    """
    sigmoid函数
    :param x: 输入矩阵
    :return: 输出矩阵
    """
    return 1 / (1 + np.exp(-x))


def plot_decision_boundary(data_set, label, theta):
    """
    绘制数据点和决策边界
    :param data_set: 数据集
    :param label: 数据标签
    :param theta: 回归系数
    :return: 图像
    """
    data_set = np.array(data_set)
    point_x = data_set[:, 1]
    point_y = data_set[:, 2]
    plt.scatter(point_x, point_y, c=label)
    x = np.linspace(-4, 4, 100)
    # 画决策边界的直线（二维）
    y = (-(theta[0] + theta[1] * x) / theta[2]).reshape((data_set.shape[0], 1))
    plt.plot(x, y)
    plt.show()


def gradient_ascent(data_set, label):
    """
    最优化算法：梯度上升
    :param data_set: 数据集
    :param label: 标签
    :return: 回归系数
    """
    data_set = np.mat(data_set)
    label = np.mat(label).T
    m, n = np.shape(data_set)
    alpha = 0.001
    theta = np.ones((n, 1))
    cycle = 500
    for i in range(cycle):
        h_theta = sigmoid(data_set * theta)
        error = label - h_theta
        theta = theta + alpha * data_set.T * error
    return theta


def stoc_gradient_ascent0(data_set, label):
    """
    随机梯度上升
    :param data_set:
    :param label:
    :return: 回归系数
    """
    m, n = np.shape(data_set)
    theta = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(data_set[i] * theta))
        error = label[i] - h
        theta = theta + alpha * error * np.array(data_set[i])
    return theta


def stoc_gradient_ascent1(data_set, label, item_num=150):
    """
    随机梯度上升（改进）
    :param data_set:
    :param label:
    :param item_num: 迭代次数
    :return: 回归系数
    """
    m, n = np.shape(data_set)
    theta = np.ones(n)
    for j in range(item_num):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (i + j + 1) + 0.01
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_set[rand_index] * theta))
            error = label[rand_index] - h
            theta = theta + alpha * error * np.array(data_set[rand_index])
            del (data_index[rand_index])
    return theta


def classify_vector(inX, theta):
    rst = sigmoid(sum(inX * theta))
    if rst >= 0.5:
        return 1
    else:
        return 0


def load_horse_data_set():
    training_set = []
    training_label = []
    test_set = []
    test_label = []
    fr = open('horseColicTraining.txt', 'r')
    file_lines = fr.readlines()
    for line in file_lines:
        line = line.strip()
        element_list = line.split('\t')
        training_set.append([float(example) for example in element_list[:-1]])
        training_label.append(float(element_list[-1]))
    fr = open('horseColicTest.txt', 'r')
    file_lines = fr.readlines()
    for line in file_lines:
        line = line.strip()
        element_list = line.split('\t')
        test_set.append([float(example) for example in element_list[:-1]])
        test_label.append(int(element_list[-1]))
    return np.array(training_set), training_label, np.array(test_set), test_label


def test(training_set, training_label, test_set, test_label):
    theta = stoc_gradient_ascent1(training_set, training_label)
    test_data_num = test_set.shape[0]
    error_count = 0
    for i in range(test_data_num):
        predict = classify_vector(test_set[i], theta)
        if predict != test_label[i]:
            error_count += 1
    print("Total number is %d, error number is %d, error rate is %.5f%%"
          % (test_data_num, error_count, error_count / test_data_num))


if __name__ == '__main__':
    data_set, label = load_data_set()
    theta = gradient_ascent(data_set, label)
    plot_decision_boundary(data_set, label, theta)
    theta2 = stoc_gradient_ascent1(data_set, label)
    plot_decision_boundary(data_set, label, theta2)
