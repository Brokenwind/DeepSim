# /usr/bin/env python
# coding=utf-8

import multiprocessing
import os
import logging
import fasttext
import gensim
import jieba_fast as jieba
from gensim.models.word2vec import LineSentence

import extract_wiki
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# 自定义的词典用于结巴分词
JIEBA_DICT_SELF_DEFINE = '../data/dict.txt'
# 存放模型的目录
MODEL_DIR = "../model/"
# 经过去除无用字符，繁体转简体后的WIKI文件
PROCESSED_WIKI_FILE_PATH = '../resources/std/'
# 蚂蚁金服的数据文件
ANT_NLP_FILE_PATH = '../data/atec_nlp_sim.csv'
# 字符级别的语料库(需要生成)
CHAR_LEVEL_CORPUS = '../data/train_char.txt'
# 单词级别的语料库
WORD_LEVEL_CORPUS = '../data/train_word.txt'


class CorpusChars(object):
    '''语料库中的每一行生成字符级别的列表
    '''

    def __init__(self):
        pass

    def __iter__(self):
        with open(ANT_NLP_FILE_PATH, "r", encoding="utf8") as atec:
            logging.info('generating char corpus, processing file %s', ANT_NLP_FILE_PATH)
            for line in atec:
                lineno, s1, s2, label = line.strip().split("\t")
                s1 = utils.remove_punctuation(s1)
                s2 = utils.remove_punctuation(s2)
                yield list(s1) + list(s2)

        for file in extract_wiki.list_all_files(PROCESSED_WIKI_FILE_PATH):
            logging.info('generating char corpus, processing file %s', file)
            with open(file, 'r', encoding="utf8") as wiki:
                for line in wiki:
                    line = utils.remove_punctuation(line)
                    if len(line) > 0:
                        yield [char for char in line if char and 0x4E00 <= ord(char[0]) <= 0x9FA5]


class CorpusWords(object):
    '''语料库中的每一行进行结巴分词
    '''

    def __init__(self):
        # 加载蚂蚁相关的字典
        jieba.load_userdict(JIEBA_DICT_SELF_DEFINE)
        pass

    def __iter__(self):
        with open(ANT_NLP_FILE_PATH, "r", encoding="utf8") as atec:
            logging.info('generating word corpus, processing file %s', ANT_NLP_FILE_PATH)
            for line in atec:
                line_code, s1, s2, label = line.strip().split("\t")
                s1 = utils.remove_punctuation(s1)
                s2 = utils.remove_punctuation(s2)
                yield list(jieba.cut(s1)) + list(jieba.cut(s2))
        for file in extract_wiki.list_all_files(PROCESSED_WIKI_FILE_PATH):
            logging.info('generating word corpus, processing file %s', file)
            with open(file, 'r', encoding="utf8") as wiki:
                for line in wiki:
                    line = utils.remove_punctuation(line)
                    if len(line) > 0:
                        # 汉字的unicode编码范围是[0x4E00,0x9FA5]
                        yield [word for word in list(jieba.cut(line)) if word and 0x4E00 <= ord(word[0]) <= 0x9FA5]


def generate_data():
    '''生成字符级别和词语级别的语料库
    :return:
    '''
    with open(CHAR_LEVEL_CORPUS, "w", encoding="utf8") as file:
        mychars = CorpusChars()
        for cs in mychars:
            file.write(" ".join(cs) + "\n")

    with open(WORD_LEVEL_CORPUS, "w", encoding="utf8") as file:
        mywords = CorpusWords()
        for ws in mywords:
            file.write(" ".join(ws) + "\n")


def train_embedding_gensim():
    '''
    使用gesim 生成字符级别和单词级别的词嵌入
    :return:
    '''
    logging.info('generating CHAR embedding %s with gensim', 'char2vec_gensim')
    dim = 256
    embedding_size = dim
    model = gensim.models.Word2Vec(LineSentence(CHAR_LEVEL_CORPUS),
                                   size=embedding_size,
                                   window=5,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    model.save(os.path.join(MODEL_DIR, "char2vec_gensim" + str(embedding_size)))
    # model.wv.save_word2vec_format("model/char2vec_org"+str(embedding_size),"model/chars"+str(embedding_size),binary=False)

    logging.info('generating WORD embedding %s with gensim', 'word2vec_gensim')
    dim = 256
    embedding_size = dim
    model = gensim.models.Word2Vec(LineSentence(WORD_LEVEL_CORPUS),
                                   size=embedding_size,
                                   window=5,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())

    model.save(os.path.join(MODEL_DIR, "word2vec_gensim" + str(embedding_size)))
    # model.wv.save_word2vec_format("model/word2vec_org"+str(embedding_size),"model/vocabulary"+str(embedding_size),binary=False)


def train_embedding_fasttext():
    '''
    使用fasttext 生成字符级别和单词级别的词嵌入
    :return:
    '''
    # Skipgram model
    logging.info('generating CHAR embedding %s with fasttext using %s algorithm', 'char2vec_fastskip256', 'Skipgram')
    model = fasttext.skipgram(CHAR_LEVEL_CORPUS, os.path.join(MODEL_DIR, 'char2vec_fastskip256'), word_ngrams=2, ws=5,
                              min_count=10, dim=256)
    del (model)

    # CBOW model
    logging.info('generating CHAR embedding %s with fasttext using %s algorithm', 'char2vec_fastcbow256', 'CBOW')
    model = fasttext.cbow(CHAR_LEVEL_CORPUS, os.path.join(MODEL_DIR, 'char2vec_fastcbow256'), word_ngrams=2, ws=5,
                          min_count=10, dim=256)
    del (model)

    # Skipgram model
    logging.info('generating WORD embedding %s with fasttext using %s algorithm', 'word2vec_fastskip256', 'Skipgram')
    model = fasttext.skipgram(WORD_LEVEL_CORPUS, os.path.join(MODEL_DIR, 'word2vec_fastskip256'), word_ngrams=2, ws=5,
                              min_count=10, dim=256)
    del (model)

    # CBOW model
    logging.info('generating WORD embedding %s with fasttext using %s algorithm', 'word2vec_fastcbow256', 'CBOW')
    model = fasttext.cbow(WORD_LEVEL_CORPUS, os.path.join(MODEL_DIR, 'word2vec_fastcbow256'), word_ngrams=2, ws=5,
                          min_count=10, dim=256)
    del (model)


if __name__ == '__main__':
    '''
    下边三个步骤可以分别执行.
    generate_data: 用于生成字符级别和词语级别的语料库,只有这一步执行完成后,后边两步才有输入数据
    train_embedding_gensim: 用gensim包分别训练一个字符嵌入和词嵌入
    train_embedding_fasttext: 用fasttext的各个算法训练字符嵌入和词嵌入
    '''
    generate_data()
    train_embedding_gensim()
    # train_embedding_fasttext()
