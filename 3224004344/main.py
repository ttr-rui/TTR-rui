import math
import sys
from collections import Counter

import jieba

from errors import PlagiarismError
from file_reader import read_text

# 关闭 jieba 加载词典时的日志输出，保持终端干净（60 高于 CRITICAL）
jieba.setLogLevel(60)

# 标点符号与空白字符不携带语义，分词后需要过滤掉
NOISE_CHARS = set(
    "，。、；：？！“”‘’（）《》〈〉【】〔〕—…·「」『』"
    ",.;:?!\"'()[]{}<>|/\\-_=+*&^%$#@~` \t\r\n\v\f\u3000"
)

# 预构建翻译表：str.translate 在 C 层一次性完成全部替换
PUNCT_TABLE = str.maketrans({char: " " for char in NOISE_CHARS})


def tokenize(text):
    """把文本切分成词，并过滤掉标点、空白等无意义的 token。

    Args:
        text: 待分词的文本。

    Returns:
        过滤后的词列表。
    """
    cleaned = text.translate(PUNCT_TABLE)
    return [
        word for word in jieba.lcut(cleaned, cut_all=False) if word.strip()
    ]


def build_word_freq(tokens):
    """统计词频，构造「词 → 出现次数」的字典。

    Args:
        tokens: 词列表。

    Returns:
        词频字典；输入为空列表时返回空字典。
    """
    return dict(Counter(tokens))


def cosine_similarity(freq_a, freq_b):
    """计算两个词频向量的余弦相似度。

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


def main():
    """程序入口：解析命令行参数、计算重复率、写入答案文件。

    Returns:
        进程退出码，0 表示成功，1 表示出错。
    """
    if len(sys.argv) != 4:
        print("用法：python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        print(f"当前收到 {len(sys.argv) - 1} 个参数，需要 3 个。")
        return 1

    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    answer_path = sys.argv[3]

    try:
        similarity = calc_similarity(original_path, copied_path)
    except PlagiarismError as error:
        print(f"错误：{error}")
        return 1

    try:
        with open(answer_path, "w", encoding="utf-8") as file:
            file.write(f"{similarity:.2f}")
    except OSError as error:
        print(f"错误：无法写入答案文件 {answer_path}（{error}）")
        return 1

    print(f"重复率 {similarity:.2f} 已写入 {answer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
