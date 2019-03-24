'''
这个模块主要是加载训练好的词向量和字符向量
'''
import time

start_time = time.time()
import os
import gensim
import logging
import re
import pandas as pd
import jieba_fast as jieba
from keras.preprocessing.sequence import pad_sequences
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split, KFold

# 下边引入自定义模块
from keras.layers import *
from constant import *
from utils import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# 加载word embedding 和 char embedding
logging.info('starting loading embedding')

if EMBEDDING_MODEL_TYPE == "gensim":
    char_embedding_model = gensim.models.Word2Vec.load(os.path.join(MODEL_DIR, "char2vec_gensim%s" % VECTOR_LENGTH))
    # 获取字符和字符的索引
    char2index = {v: k for k, v in enumerate(char_embedding_model.wv.index2word)}
    word_embedding_model = gensim.models.Word2Vec.load(os.path.join(MODEL_DIR, "word2vec_gensim%s" % VECTOR_LENGTH))
    # 获取词语和词语的索引
    word2index = {v: k for k, v in enumerate(word_embedding_model.wv.index2word)}

elif EMBEDDING_MODEL_TYPE == "fastskip" or EMBEDDING_MODEL_TYPE == "fastcbow":
    char_fastcbow = FastText.load_fasttext_format(
        os.path.join(MODEL_DIR, "char2vec_%s%d.bin" % (EMBEDDING_MODEL_TYPE, VECTOR_LENGTH)), full_model=False)
    char_embedding_matrix = char_fastcbow.wv.vectors
    char2index = {v: k for k, v in enumerate(char_fastcbow.wv.index2word)}
    word_fastcbow = FastText.load_fasttext_format(
        os.path.join(MODEL_DIR, "word2vec_%s%d.bin" % (EMBEDDING_MODEL_TYPE, VECTOR_LENGTH)), full_model=False)
    word_embedding_matrix = word_fastcbow.wv.vectors
    word2index = {v: k for k, v in enumerate(word_fastcbow.wv.index2word)}
logging.info('end loading embedding')


def get_embedding_layers(dtype, input_length, w2v_length, with_weight=True):
    '''
    生成keras的embedding layer
    :param dtype: word/char/all
    :param input_length: 输入句子的的(词语/字符)的个数
    :param w2v_length: (词/字符)嵌入向量维度
    :param with_weight: 是否使用训练好的embedding模型初始化keras的Embedding层
    :return:
    '''

    def __get_embedding_layers(dtype, input_length, w2v_length, with_weight=True):
        if dtype == 'word':
            embedding_length = len(word2index)
        elif dtype == 'char':
            embedding_length = len(char2index)

        if with_weight:
            if EMBEDDING_MODEL_TYPE == "gensim":
                # gensim生成的word embedding 可以直接获取keras的Embedding层
                if dtype == 'word':
                    embedding = word_embedding_model.wv.get_keras_embedding(train_embeddings=True)
                else:
                    embedding = char_embedding_model.wv.get_keras_embedding(train_embeddings=True)

            elif EMBEDDING_MODEL_TYPE == "fastskip" or EMBEDDING_MODEL_TYPE == "fastcbow":
                if dtype == 'word':
                    embedding = Embedding(embedding_length, w2v_length, input_length=input_length,
                                          weights=[word_embedding_matrix], trainable=True)
                else:
                    embedding = Embedding(embedding_length, w2v_length, input_length=input_length,
                                          weights=[char_embedding_matrix], trainable=True)
        else:
            embedding = Embedding(embedding_length, w2v_length, input_length=input_length, trainable=True)

        return embedding

    if dtype == "both":
        embedding = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            embedding.append(__get_embedding_layers(dtype, input_length, w2v_length, with_weight))
        return embedding
    else:
        return __get_embedding_layers(dtype, input_length, w2v_length, with_weight)


def load_data(dtype="both", input_length=[20, 24], w2v_length=VECTOR_LENGTH):
    '''
    读取输入文件,获取每个句子的中的词语的索引或者是字符的索引
    :param dtype:
    :param input_length:
    :param w2v_length:
    :return: 返回是一个list,每一个元素也是list, 表示一个句子的字符/词语的索引列表
    '''

    def __load_data(dtype="word", input_length=20, w2v_length=VECTOR_LENGTH):
        filename = os.path.join(MODEL_DIR, "%s_%d_%d" % (dtype, input_length, w2v_length))
        if os.path.exists(filename):
            return pd.read_pickle(filename)

        data_left_sentence = []
        data_right_sentence = []
        labels = []
        for line in open(ANT_NLP_FILE_PATH, "r", encoding="utf8"):
            line_number, sentence1, sentence2, label = line.strip().split("\t")
            # 句子中出现连续*号,表示数字
            star = re.compile("\*+")
            sentence1 = remove_punctuation(star.sub("1", sentence1))
            sentence2 = remove_punctuation(star.sub("1", sentence2))
            if dtype == "word":
                data_left_sentence.append(
                    [word2index[word] for word in list(jieba.cut(sentence1)) if word in word2index])
                data_right_sentence.append(
                    [word2index[word] for word in list(jieba.cut(sentence2)) if word in word2index])
            if dtype == "char":
                data_left_sentence.append([char2index[char] for char in sentence1 if char in char2index])
                data_right_sentence.append([char2index[char] for char in sentence2 if char in char2index])
            labels.append(int(label))

        logging.info('length of featured sentence is ' + str(len(data_left_sentence)))
        # 对齐语料中句子的长度
        data_left_sentence = pad_sequences(data_left_sentence, maxlen=input_length)
        data_right_sentence = pad_sequences(data_right_sentence, maxlen=input_length)
        labels = np.array(labels)

        pd.to_pickle((data_left_sentence, data_right_sentence, labels), filename)

        return (data_left_sentence, data_right_sentence, labels)

    if dtype == "both":
        combined_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_left_sentence, data_right_sentence, labels = __load_data(dtype, input_length, w2v_length)
            combined_array.append(np.asarray(data_left_sentence))
            combined_array.append(np.asarray(data_right_sentence))
        combined_array.append(labels)
        return combined_array
    else:
        return __load_data(dtype, input_length, w2v_length)


def input_data(sentence1, sentence2, dtype="both", input_length=[20, 24]):
    def __input_data(sentence1, sentence2, dtype="word", input_length=20):
        data_left_sentence = []
        data_right_sentence = []
        for s1, s2 in zip(sentence1, sentence2):
            if dtype == "word":
                # 句子中出现连续*号,表示数字
                star = re.compile("\*+")
                data_left_sentence.append(
                    [word2index[word] for word in list(jieba.cut(star.sub("1", s1))) if word in word2index])
                data_right_sentence.append(
                    [word2index[word] for word in list(jieba.cut(star.sub("1", s2))) if word in word2index])
            if dtype == "char":
                data_left_sentence.append([char2index[char] for char in s1 if char in char2index])
                data_right_sentence.append([char2index[char] for char in s2 if char in char2index])

        # 对齐语料中句子的长度
        data_left_sentence = pad_sequences(data_left_sentence, maxlen=input_length)
        data_right_sentence = pad_sequences(data_right_sentence, maxlen=input_length)

        return [data_left_sentence, data_right_sentence]

    if dtype == "both":
        ret_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_left_sentence, data_right_sentence = __input_data(sentence1, sentence2, dtype, input_length)
            ret_array.append(data_left_sentence)
            ret_array.append(data_right_sentence)
        return ret_array
    else:
        return __input_data(sentence1, sentence2, dtype, input_length)


def split_data(data, mode="train", test_size=test_size, random_state=random_state):
    '''
    训练集和测试集合\划分
    :param data:
      mode == "train":  划分成用于训练的四元组
      mode == "orig":   划分成两组数据
    :param mode:
    :param test_size:
    :param random_state:
    :return:
    '''

    train = []
    test = []
    for data_i in data:
        # 快速调试
        if FAST_MODE:
            data_i, _ = train_test_split(data_i, test_size=1 - FAST_RATE, random_state=random_state)
        train_data, test_data = train_test_split(data_i, test_size=test_size, random_state=random_state)
        train.append(np.asarray(train_data))
        test.append(np.asarray(test_data))

    if mode == "orig":
        return train, test

    train_x, train_y, test_x, test_y = train[:-1], train[-1], test[:-1], test[-1]

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    '''
    下边语句测试使用
    '''
    # embedding = get_embedding_layers('fastskip', 20, VECTOR_LENGTH)
    # print(type(embedding))
    # (data_left_sentence, data_right_sentence, labels) = load_data('word', 20)
    # print(data_left_sentence[0].shape)
