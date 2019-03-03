import time

start_time = time.time()
from constant import *
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
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, Callback, ReduceLROnPlateau, \
    LearningRateScheduler
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from keras.regularizers import L1L2, l2
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
import copy

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
    char_fastcbow = FastText.load(os.path.join(MODEL_DIR, "char2vec_%s%d" % (EMBEDDING_MODEL_TYPE, VECTOR_LENGTH)))
    char_embedding_matrix = char_fastcbow.wv.vectors
    char2index = {v: k for k, v in enumerate(char_fastcbow.wv.index2word)}
    word_fastcbow = FastText.load(os.path.join(MODEL_DIR, "word2vec_%s%d" % (EMBEDDING_MODEL_TYPE, VECTOR_LENGTH)))
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


def get_model(cfg, model_weights=None):
    print("=======   CONFIG: ", cfg)

    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
    embedding = get_embedding_layers(dtype, input_length, w2v_length, with_weight=True)

    if model_type == "esim":
        model = esim(pretrained_embedding=embedding,
                     maxlen=input_length,
                     lstm_dim=300,
                     dense_dim=300,
                     dense_dropout=0.5)
    elif model_type == "decom":
        model = decomposable_attention(pretrained_embedding=embedding,
                                       projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                                       compare_dim=500, compare_dropout=0.2,
                                       dense_dim=300, dense_dropout=0.2,
                                       lr=1e-3, activation='elu', maxlen=input_length)
    elif model_type == "siamese":
        model = siamese(pretrained_embedding=embedding, input_length=input_length, w2v_length=w2v_length,
                        n_hidden=n_hidden)
    elif model_type == "dssm":
        model = DSSM(pretrained_embedding=embedding, input_length=input_length, lstmsize=90)

    if model_weights is not None:
        model.load_weights(model_weights)

    # keras.utils.plot_model(model, to_file=model_dir+model_type+"_"+dtype+'.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model


if __name__ == '__main__':
    embedding = get_embedding_layers('fastskip', 20, VECTOR_LENGTH)
    print(type(embedding))
