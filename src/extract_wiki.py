import codecs
import os
import re

# from gensim.corpora import WikiCorpus
from opencc import OpenCC

# 原始wiki文件
ORIGIN_WIKI_FILE_PATH = '../resources/zhwiki-latest-pages-articles.xml.bz2'
# wiki信息提取后的文件
EXTRACTED_WIKI_FILE_PATH = '../resources/zhwiki/'
# 经过去除无用字符，繁体转简体后的文件
PROCESSED_WIKI_FILE_PATH = '../resources/std/'


def extract_wiki(file_path):
    os.system('WikiExtractor.py -b 500M -o zhwiki  ' + file_path)


def remove_signs(input_file, output_file):
    '''去掉无用的标点
    :param input_file:
    :return:
    '''
    opencc = OpenCC("t2s")
    # doc标签以及空白行
    p0 = re.compile(r'<doc.*>|</doc>|^$|^\s$')
    # 存在这样的：-{H|zh-hant:心裡學;zh-hans:心里学;}-,是标题
    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[（\(][，；。？！\s]*[）\)]')
    # 这种符号都替换为中文引号
    p3 = re.compile(r'[「『]')
    # 这种符号都替换为中文引号
    p4 = re.compile(r'[」』]')
    # 英文标点符号,都需要去除
    p5 = re.compile(r"[ `~!@#$%^&*\(\)\-_=+\[\]\{\}\\\|;:\'\",<.>/?a-zA-Z\d]+")
    # 引号或者括号中为空，或者标点符号之间为空，都需要去除
    p6 = re.compile("(（）)|(“”)|(「」)|(《》)|(“”)|(‘’)|(【】)|[，。？——！]{2,}")

    # (file_path, file_name) = os.path.split(input_file)
    outfile = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            # 丢弃空白行以及含有<doc>标签的行
            if re.match(p0, line):
                continue
            # \2 的意思正则表达式的第几个moudle
            line = p1.sub(r'\2', line)
            line = p1.sub(r'', line)
            line = p2.sub(r'', line)
            line = p3.sub(r'“', line)
            line = p4.sub(r'”', line)
            line = p5.sub(r'', line)
            line = p6.sub(r'', line)

            # 繁体转简体
            line = opencc.convert(line)
            outfile.write(line)
    outfile.close()


def remove_unusable_all(input_path, output_path):
    '''
    去除经过WikiExtractor提取后的无用符号,并输出到新文件夹
    :param input_path:
    :return:
    '''
    # 获取绝对目录
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise RuntimeError("Not found input_path")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # 处理每一个文件,新生成的文件与旧文件同名
    for cur_file in list_all_files(input_path):
        (file_path, file_name) = os.path.split(cur_file)
        remove_signs(cur_file, os.path.join(output_path, file_name))


def list_all_files(rootdir):
    '''
    # 列出文件夹下所有文件,包括子目录
    :param rootdir:
    :return:
    '''
    _files = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)

    return _files


if __name__ == '__main__':
    # extract_wiki(FILE_PATH)
    remove_unusable_all(EXTRACTED_WIKI_FILE_PATH, PROCESSED_WIKI_FILE_PATH)
