from math import log
import pickle


def create_data():
    """
    产生数据
    :rtype: 数据列表，特征列表
    """
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    feature = ['no surfacing', 'flippers']
    return data_set, feature


def shanno(data_set):
    """
    计算标签数组的香农熵
    :param data_set: 输入数据，列表形式
    :return: 香农熵
    """
    size = len(data_set)
    label_count = {}
    for data in data_set:
        label = data[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    shanno_rst = 0
    for i in label_count.values():
        po = i / size
        shanno_rst -= po * log(po, 2)
    return shanno_rst


def split_data(data_set, axis, value):
    """
    依据某项特征分割数据
    :param data_set: 数据列表
    :param axis: 特征下标
    :param value: 特征值
    :return: 满足条件的数据（已去除选择的特征值）
    """
    rst_data = []
    for data in data_set:
        tmp_data = []
        if data[axis] == value:
            tmp_data.extend(data[:axis])
            tmp_data.extend(data[axis + 1:])
            rst_data.append(tmp_data)
    return rst_data


def choose_best_feature(data_set):
    """
    为数据集选择最好的分类特征
    :param data_set: 输入的数据集，包括特征值和标签
    :return: 分类效果最好的特征下标
    """
    # 数据集特征数量
    feature_num = len(data_set[0]) - 1
    # 基准香农熵
    base_shanno = shanno(data_set)
    # 最大的信息增益
    max_gain = 0
    # 信息增益最大的特征
    best_feature = -1
    for i in range(feature_num):
        # 找到该特征中的所有值，用set存储，得到unique features
        feature_list = [example[i] for example in data_set]
        feature_set = set(feature_list)
        # 获得按此特征切分数据的香农熵
        sub_shanno_sum = 0
        for feature_value in feature_set:
            sub_data_set = split_data(data_set, i, feature_value)
            po = len(sub_data_set) / len(data_set)
            sub_shanno_sum += po * shanno(sub_data_set)
        # 得出信息增量，并判断信息增量是否为最大
        info_gain = base_shanno - sub_shanno_sum
        if info_gain > max_gain:
            max_gain = info_gain
            best_feature = i
    return best_feature


def majority(label_list):
    """
    没有特征值可供选择时选出出现次数最多的标签
    :param label_list: 标签列表
    :return: 出现次数最多的标签
    """
    label_count = {}
    for label in label_list:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    sorted_count = sorted(label_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_count[0][0]


def create_decision_tree(data_set, feature_list):
    """
    create the decision tree (recursion)
    :param data_set: input data(array)
    :param feature_list: label_name_list(list)
    :return:
    """
    """
    递归边界（叶子节点）：
    1 label_list all the same
    2 data_set has no feature
    """
    label_list = [example[-1] for example in data_set]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    if len(data_set[0]) == 1:
        return majority(label_list)
    # 递归体
    best_feature = choose_best_feature(data_set)
    best_feature_text = feature_list[best_feature]
    my_tree = {best_feature_text: {}}
    best_feature_list = [example[best_feature] for example in data_set]
    best_feature_set = set(best_feature_list)
    copy_feature_list = feature_list[:]
    del (copy_feature_list[best_feature])
    for feature_value in best_feature_set:
        sub_data_set = split_data(data_set, best_feature, feature_value)
        my_tree[best_feature_text][feature_value] = create_decision_tree(sub_data_set, copy_feature_list)
    return my_tree


def test(tree, features, test_feature):
    """
    根据给出的特征测试标签
    :param tree: 训练好的决策树
    :param features: 已知的特征
    :param test_feature: 测试数据
    :return: 结果标签
    """
    # 当前划分特征
    feature = list(tree.keys())[0]
    # 按当前特征划分的子树
    second_dict = tree[feature]
    feature_index = features.index(feature)
    # 遍历该特征的所有值,寻找与测试数据该特征相等的分支
    for feature_value in second_dict.keys():
        if test_feature[feature_index] == feature_value:
            # 若分支为字典型 则继续递归寻找 否则返回值（到达叶子节点）
            if type(second_dict[feature_value]).__name__ == 'dict':
                label = test(second_dict[feature_value], features, test_feature)
            else:
                label = second_dict[feature_value]
    return label


def store_tree(input_tree, filename):
    """
    存储训练完成的决策树
    :param input_tree: 输入树
    :param filename: 存入文件的名称
    :return: NULL
    """
    fr = open(filename, 'wb')
    pickle.dump(input_tree, fr)
    fr.close()


def grab_tree(filename):
    """
    读取文件中存储的树
    :param filename: 文件名
    :return: 决策树
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    data_set, feature = create_data()
    tree = create_decision_tree(data_set, feature)
    store_tree(tree, 'tree.txt')

    label = test(tree, feature, [0, 1])
    print(label)
    tree = grab_tree('tree.txt')
    print(tree)
