"""文本处理模块。

负责把原始文本转换成用于比对的「特征向量」：先用正则剔除标点、
空白等噪声字符，只保留汉字与字母数字，再按连续 N 个字符滑动取窗
切成 N-gram 特征，最后统计每个特征的出现次数。

为什么用 N-gram 而不是分词后的词袋：
    词袋模型只统计词的出现次数，完全忽略词序。对于「把相邻两字互换
    位置」这类抄袭（例如 dis 系列样例），每个词的出现次数几乎不变，
    词袋模型会把明显被改动的文本误判为高度重复。改用 2-gram（连续
    两个字符组成一个特征）后，特征本身携带了局部顺序信息：乱序会
    破坏掉大量 2-gram，相似度随之下降，判定结果更符合直觉。
"""

import re
from collections import Counter

# N-gram 的窗口大小：2 表示「连续两个字符」构成一个特征
NGRAM_SIZE = 2

# 只保留汉字与字母数字；标点、空白等噪声字符统统一并剔除
NOISE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(text):
    """把文本切分成 N-gram 特征，返回特征生成器。

    先用正则一次性剔除标点与空白，再对保留下来的字符序列按
    NGRAM_SIZE 滑动取窗；以生成器形式返回，调用方可以边取特征边
    统计，无需在内存中落地完整的特征列表。

    Args:
        text: 待处理的文本。

    Returns:
        剔除噪声之后的 N-gram 特征生成器；文本有效字符不足
        NGRAM_SIZE 个时返回空生成器。
    """
    cleaned = NOISE_PATTERN.sub("", text)
    return (
        cleaned[i:i + NGRAM_SIZE]
        for i in range(len(cleaned) - NGRAM_SIZE + 1)
    )


def build_word_freq(tokens):
    """统计特征频率，构造「特征 → 出现次数」的映射。

    直接消费特征迭代器，不产生中间列表。

    Args:
        tokens: 特征的可迭代对象。

    Returns:
        特征计数（Counter，dict 的子类）；输入为空时返回空 Counter。
    """
    return Counter(tokens)
