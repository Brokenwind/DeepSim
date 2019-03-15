import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from extract_wiki import list_all_files

LOG_TITLES = ['epoch', 'time', 'loss', 'entropy', 'acc', 'val_loss', 'val_entropy', 'val_acc', 'type']
ALGORITHM_TYPES = ['decom_char', 'decom_word', 'dssm_both', 'esim_char', 'esim_word', 'siamese_char', 'siamese_word']
EMBEDDING_TYPES = ['fastcbow', 'fastskip']
STATISTIC_TITLES = ['embedding_type', 'algorithm_type', 'epochs', 'epoch_time', 'parameters', 'test_acc']


def load_data(input_path, types, titles):
    '''
    加载处理后的日志数据
    :param input_path: 日志所在的文件夹
    :param types: all_types的子集,需要的算法类型
    :return:
    '''
    # 目录的时候,获取目录中所有文件
    if os.path.isdir(input_path):
        data_list = []
        files = list_all_files(input_path)
        for file in files:
            base_dir, suffix = os.path.splitext(file)
            if suffix == '.csv':
                data_list.append(pd.read_csv(file, names=titles, header=0))
        data = pd.concat(data_list)
    # 文件则直接读取
    else:
        data = pd.read_csv(input_path, names=titles, header=0)
    # 如果types的值不为空,就按照数据中的type进行过滤
    if types:
        data = data[data['type'].isin(types)]

    return data


def plot_data(embedding_type, value, types):
    '''
    默认使用 epoch 作为横坐标,纵坐标由value指定
    :param embedding_type: 词嵌入算法类型, fastcbow/fastskip
    :param value: 所选择的纵坐标,从titles中挑选一个值
    :param types: 需要画折线图的算法类型, 从all_types中选择需要绘图的列表
    :return:
    '''
    if embedding_type == 'fastcbow':
        input_path = '../logs/fastcbow/'
    else:
        input_path = '../logs/fastskip/'
    data = load_data(input_path, types, LOG_TITLES)
    sns.pointplot(x="epoch", y=value, hue="type", data=data)
    plt.show()


def plot_loss(embedding_type):
    '''
    损失函数折线图
    :param embedding_type: 词嵌入算法,fastcbow/fastskip
    :return:
    '''
    # types = ['esim_char','esim_word']
    # types = ['decom_char','decom_word']
    # types = ['siamese_char', 'siamese_word']
    types = ['esim_char', 'esim_word', 'decom_char', 'decom_word']
    plot_data(embedding_type, 'loss', types)


def plot_acc(embedding_type):
    '''
    在训练集上的精确度
    :param embedding_type: 词嵌入算法,fastcbow/fastskip
    :return:
    '''
    # types = ['esim_char','esim_word']
    # types = ['decom_char','decom_word']
    # types = ['siamese_char', 'siamese_word']
    types = ['esim_char', 'esim_word', 'decom_char', 'decom_word']
    plot_data(embedding_type, 'acc', types)


def plot_val_acc(embedding_type):
    '''
    在验证集上的精确度
    :param embedding_type: 词嵌入算法,fastcbow/fastskip
    :return:
    '''
    # types = ['esim_char','esim_word']
    # types = ['decom_char','decom_word']
    # types = ['siamese_char', 'siamese_word']
    types = ['esim_char', 'esim_word', 'decom_char', 'decom_word']
    plot_data(embedding_type, 'val_acc', types)


def plot_bar(embedding_type, tile):
    '''
    根据statistics.csv中的数据画柱状图,横坐标已经确定用algorithm_type
    :param embedding_type: 词嵌入类型,这是一个list类型,值是EMBEDDING_TYPES的子集
    :param tile: 柱状图的纵坐标
    :return:
    '''
    data = load_data('../logs/statistics.csv', None, STATISTIC_TITLES)
    data = data[data['embedding_type'].isin(embedding_type)]
    sns.barplot(x='algorithm_type', y=tile, data=data)
    plt.show()


def plot_epochs(embedding_type):
    '''
    执行的轮次柱状图
    :param embedding_type:
    :return:
    '''
    plot_bar(embedding_type, 'epochs')

def plot_epoch_time(embedding_type):
    '''
    每个轮次执行时间
    :param embedding_type:
    :return:
    '''
    plot_bar(embedding_type, 'epoch_time')

def plot_parameters(embedding_type):
    '''
    参数个数
    :param embedding_type:
    :return:
    '''
    plot_bar(embedding_type, 'parameters')

def plot_test_acc(embedding_type):
    '''
    测试准确率
    :param embedding_type:
    :return:
    '''
    plot_bar(embedding_type, 'test_acc')

if __name__ == '__main__':
    # plot_val_acc('fastskip')
    # plot_epochs(['fastcbow'])
    # plot_epoch_time(['fastcbow'])
    # plot_parameters(['fastcbow'])
    plot_test_acc(['fastcbow'])

