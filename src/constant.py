'''
这里定义项目用到的常量
'''

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
# 字符向量或者词嵌入向量维度
VECTOR_LENGTH = 256
# 选取的词嵌入算法
EMBEDDING_MODEL_TYPE = 'gensim'
