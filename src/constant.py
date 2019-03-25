import os

'''
这里定义项目用到的常量
'''

###################################################
# 数据相关参数
###################################################
# 原始wiki文件
ORIGIN_WIKI_FILE_PATH = '../resources/zhwiki-latest-pages-articles.xml.bz2'
# wiki信息提取后的文件
EXTRACTED_WIKI_FILE_PATH = '../resources/zhwiki/'
# 经过去除无用字符，繁体转简体后的文件
PROCESSED_WIKI_FILE_PATH = '../resources/std/'
# 自定义的词典用于结巴分词
JIEBA_DICT_SELF_DEFINE = '../data/dict.txt'
# 存放模型的目录
MODEL_DIR = "../model/"
# 蚂蚁金服的数据文件
ANT_NLP_FILE_PATH = '../data/atec_nlp_sim.csv'
# 字符级别的语料库(需要生成)
CHAR_LEVEL_CORPUS = '../data/train_char.txt'
# 单词级别的语料库
WORD_LEVEL_CORPUS = '../data/train_word.txt'

###################################################
# 词向量相关参数
###################################################
# 字符向量或者词嵌入向量维度
VECTOR_LENGTH = 256
# 选取的词嵌入算法
EMBEDDING_MODEL_TYPE = 'gensim'

###################################################
# 训练相关参数
###################################################
MAX_LEN = 30
MAX_EPOCH = 90
train_batch_size = 64
test_batch_size = 500
earlystop_patience, plateau_patience = 8, 2
test_size = 0.025
random_state = 42
# 快速调试，其评分不作为参考
FAST_MODE, FAST_RATE = False, 0.01
CONFIG_PATH = os.path.join(MODEL_DIR, "all_configs.json")

cfgs = [
    ("siamese", "char", 24, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [100, 80, 64], 102 - 5, earlystop_patience),  # 69s
    ("siamese", "word", 20, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [100, 80, 64], 120 - 4, earlystop_patience),  # 59s
    ("esim", "char", 24, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [], 18, earlystop_patience),  # 389s
    ("esim", "word", 20, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [], 21, earlystop_patience),  # 335s
    ("decom", "char", 24, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [], 87 - 2, earlystop_patience),  # 84s
    ("decom", "word", 20, EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [], 104 - 4, earlystop_patience),  # 71s
    #("dssm", "both", [20, 24], EMBEDDING_MODEL_TYPE, VECTOR_LENGTH, [], 124 - 8, earlystop_patience),  # 55s
]
