"""相似度计算模块。

负责在特征向量之上计算余弦相似度，并对外提供「给出两个文件路径、
返回重复率」的编排入口，是入口层与文本处理层之间的桥梁。
"""

import math

from file_reader import read_text
from text_processor import build_word_freq, tokenize


def cosine_similarity(freq_a, freq_b):
    """计算两个特征向量的余弦相似度。

    cos(theta) = A·B / (|A| * |B|)，只遍历两篇文章共有的词来计算内积，
    避免遍历整个词表，这是稀疏向量的常见优化。

    Args:
        freq_a: 第一篇文章的词频字典。
        freq_b: 第二篇文章的词频字典。

    Returns:
        0~1 之间的相似度；任一方没有有效词时返回 0.0。
    """
    if not freq_a or not freq_b:
        return 0.0

    # 键集合的交集就是两篇文章共有的词
    common_words = freq_a.keys() & freq_b.keys()
    dot_product = sum(
        freq_a[word] * freq_b[word] for word in common_words
    )

    # 模长各算一次并复用，避免在循环中重复开方
    norm_a = math.sqrt(sum(count * count for count in freq_a.values()))
    norm_b = math.sqrt(sum(count * count for count in freq_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def calc_similarity(original_path, copied_path):
    """计算两个文件所载文章的重复率。

    依次完成：读取原文与抄袭版 → 切分 2-gram 特征 → 统计特征频率
    → 计算余弦相似度。

    Args:
        original_path: 原文文件路径。
        copied_path: 抄袭版论文文件路径。

    Returns:
        0~1 之间的重复率。

    Raises:
        FileReadError: 任一文件无法读取时抛出。
    """
    original_tokens = tokenize(read_text(original_path))
    copied_tokens = tokenize(read_text(copied_path))
    return cosine_similarity(
        build_word_freq(original_tokens),
        build_word_freq(copied_tokens),
    )
