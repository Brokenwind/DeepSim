import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from extract_wiki import list_all_files

TITLES = ['epoch', 'time', 'loss', 'entropy', 'acc', 'val_loss', 'val_entropy', 'val_acc', 'type']
ALL_TYPES = ['decom_char', 'decom_word', 'dssm_both', 'esim_char', 'esim_word', 'siamese_char', 'siamese_word']


def load_data(input_path, types):
    '''
    加载处理后的日志数据
    :param input_path: 日志所在的文件夹
    :param types: all_types的子集,需要的算法类型
    :return:
    '''
    if os.path.isdir(input_path):
        data_list = []
        files = list_all_files(input_path)
        for file in files:
            base_dir, suffix = os.path.splitext(file)
            if suffix == '.csv':
                data_list.append(pd.read_csv(file, names=TITLES, header=0))
        data = pd.concat(data_list)
    else:
        data = pd.read_csv(input_path, names=TITLES, header=0)
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
    data = load_data(input_path, types)
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


if __name__ == '__main__':
    plot_val_acc('fastskip')
