import math
import sys
from collections import Counter

import jieba

jieba.setLogLevel(60)

NOISE_CHARS = set(
    "，。、；：？！“”‘’（）《》〈〉【】〔〕—…·「」『』"
    ",.;:?!\"'()[]{}<>|/\\-_=+*&^%$#@~` \t\r\n\v\f\u3000"
)

# 预构建翻译表：str.translate 在 C 层一次性完成全部替换
PUNCT_TABLE = str.maketrans({char: " " for char in NOISE_CHARS})


class PlagiarismError(Exception):
    """本程序所有自定义异常的基类，便于在入口处统一捕获。"""


class ParameterError(PlagiarismError):
    """命令行参数的个数不正确。"""


class FileReadError(PlagiarismError):
    """输入文件不存在、无法读取或内容为空。"""


class FileWriteError(PlagiarismError):
    """答案文件无法写入。"""


def read_text(path):
    """读取文本文件内容。"""
    text = None
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as file:
                text = file.read()
            break
        except FileNotFoundError:
            raise FileReadError(f"找不到文件：{path}")
        except UnicodeDecodeError:
            continue
        except OSError as error:
            raise FileReadError(f"无法读取文件：{path}（{error}）")

    if text is None:
        try:
            with open(path, "rb") as file:
                text = file.read().decode("utf-8", errors="ignore")
        except OSError as error:
            raise FileReadError(f"无法读取文件：{path}（{error}）")

    if not text.strip():
        raise FileReadError(f"文件内容为空：{path}")

    return text


def tokenize(text):
    """把文本切分成词，并过滤掉标点、空白等无意义的 token。

    优化点：先用 str.translate 把标点替换成空格，再做分词，
    分词结果里的标点已变成空白，过滤只需一次 strip()。

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
    """统计词频，构造「词 → 出现次数」的字典。"""
    return dict(Counter(tokens))


def cosine_similarity(freq_a, freq_b):
    """计算两个词频向量的余弦相似度。"""
    if not freq_a or not freq_b:
        return 0.0

    common_words = freq_a.keys() & freq_b.keys()
    dot_product = sum(
        freq_a[word] * freq_b[word] for word in common_words
    )
    norm_a = math.sqrt(sum(count * count for count in freq_a.values()))
    norm_b = math.sqrt(sum(count * count for count in freq_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def calc_similarity(original_path, copied_path):
    """计算两个文件所载文章的重复率。"""
    original_tokens = tokenize(read_text(original_path))
    copied_tokens = tokenize(read_text(copied_path))
    return cosine_similarity(
        build_word_freq(original_tokens),
        build_word_freq(copied_tokens),
    )


def main():
    """程序入口：解析命令行参数、计算重复率、写入答案文件。"""
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


if __name__ == "__main__":
    sys.exit(main())
