import time

start_time = time.time()
form
constant
import *
import multiprocessing
import os
import re
import json
import gensim
import logging
import jieba
import keras
import keras.backend as K
import numpy as np
import pandas as pd

from gensim.models.fasttext import FastText

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def load_embedding(vector_len, model_type):
    '''
    获取 字符嵌入 的 字符索引 和 词向量 的 词语索引
    例如词向量的词语索引:
    [...
      '属于': 987
      '能换': 988
    ...]
    :param vector_len: 模型中每个向量的维度
    :param model_type: 模型类别
    :return:
    '''
    # w2v_length = 300
    # ebed_type = "gensim"
    # ebed_type = "fastcbow"
    logging.info('start loading embedding')
    word2index = []
    char2index = []
    if model_type == "gensim":
        char_embedding_model = gensim.models.Word2Vec.load(os.path.join(MODEL_DIR, "char2vec_gensim%s" % vector_len))
        # 获取字符和字符的索引
        char2index = {v: k for k, v in enumerate(char_embedding_model.wv.index2word)}
        word_embedding_model = gensim.models.Word2Vec.load(os.path.join(MODEL_DIR, "word2vec_gensim%s" % vector_len))
        # 获取词语和词语的索引
        word2index = {v: k for k, v in enumerate(word_embedding_model.wv.index2word)}

    elif model_type == "fastskip" or model_type == "fastcbow":
        char_fastcbow = FastText.load(os.path.join(MODEL_DIR, "char2vec_%s%d" % (model_type, vector_len)))
        char_embedding_matrix = char_fastcbow.wv.vectors
        char2index = {v: k for k, v in enumerate(char_fastcbow.wv.index2word)}
        word_fastcbow = FastText.load(os.path.join(MODEL_DIR, "word2vec_%s%d" % (model_type, vector_len)))
        word_embedding_matrix = word_fastcbow.wv.vectors
        word2index = {v: k for k, v in enumerate(word_fastcbow.wv.index2word)}
    logging.info('end loading embedding')

    return word2index, char2index


if __name__ == '__main__':
    word2index, char2index = load_embedding(256, 'fastskip')
