# coding: utf-8

import re
import logging
import HTMLParser

logger = logging.getLogger(__file__)

html_parser = HTMLParser.HTMLParser()


def clean_txt(txt):
    """
    清洗原始文本
    1. 还原html转义字符
    2. 清除特殊符号
    :param txt:
    :return:
    """

    txt = txt.strip()

    # 还原html转义字符，&hellip; => ……
    txt = html_parser.unescape(txt)
    txt = txt.replace(u'...........', '')

    # Todo HTMLParser不能讲&#039;转换，而单独测试时是可以的，不知为何。。
    txt = txt.replace('&#039;', '\'')

    return txt


def extract_standard_sentences(line):
    """
    提取规范句，规范句的形式如下：
    1. xxx，xxx，xxx。 => 完整提取
    2. xxx, xxx =>提取后加'。'
    :param line:
    :return: 规范句集合
    """
    _SENT_CHAR_SET = u'[0-9a-zA-Z\u4e00-\u9fa5_-]'
    _rule1 = re.compile(u'((?:%s+，)*%s+[。！；？?])' % (_SENT_CHAR_SET, _SENT_CHAR_SET))
    _rule2 = re.compile(u'((?:%s+，)*%s+)' % (_SENT_CHAR_SET, _SENT_CHAR_SET))

    sents = set()

    # 1. xxx，xxx，xxx。 => 完整提取
    groups = re.findall(_rule1, line)
    for sent in groups:
        sents.add(sent)

    # 2. xxx, xxx =>提取后加'。
    if not groups:
        groups = re.findall(_rule2, line)
        for sent in groups:
            sents.add('%s。' % sent)

    return sents


def is_meaningful(sent):
    """
    判断句子是否有意义
    1. 必须至少包含两个连续的中文（一个词）
    2. 必须包含至少两个不同的字
    3. 不是无意义的评论
    :param sent:
    :return:
    """

    # 必须至少包含两个连续的中文（一个词）
    if not re.match(r'.*[\u4e00-\u9fa5]{2,999}.*', sent):
        return False

    words = re.findall(r'[\u4e00-\u9fa5]', sent)

    # 必须包含至少两个不同的字
    if len(set(words)) < 2:
        return False

    # 不是无意义的评论
    if ''.join(words) in SYS_COMMENTS:
        return False

    return True


SYS_COMMENTS = [u'此用户没有填写评论', u'好评']