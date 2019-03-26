import re
import os

def remove_punctuation(line, strip_all=True):
    '''
    去掉中文中的标点符号
    :param line:
    :param strip_all:
    :return:
    '''
    if strip_all:
        # 去除所有半角全角符号，只留字母、数字、中文
        rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
        line = rule.sub('', line)
    else:
        # 去掉手工指定这些标点符号
        punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        re_punctuation = "[{}]+".format(punctuation)
        line = re.sub(re_punctuation, "", line)
    return line


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
