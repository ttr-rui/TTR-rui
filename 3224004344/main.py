"""论文查重程序。

从命令行接收三个绝对路径参数（原文文件、抄袭版论文文件、答案文件），
计算两篇文章的重复率，并将结果写入答案文件。

用法：
    python main.py [原文文件] [抄袭版论文的文件] [答案文件]
"""

import math
import sys
from collections import Counter

import jieba

# 关闭 jieba 加载词典时的日志输出，保持终端干净（60 高于 CRITICAL）
jieba.setLogLevel(60)

# 标点符号与空白字符不携带语义，分词后需要过滤掉
NOISE_CHARS = set(
    "，。、；：？！“”‘’（）《》〈〉【】〔〕—…·「」『』"
    ",.;:?!\"'()[]{}<>|/\\-_=+*&^%$#@~` \t\r\n\v\f\u3000"
)


class PlagiarismError(Exception):
    """本程序所有自定义异常的基类，便于在入口处统一捕获。"""


class ParameterError(PlagiarismError):
    """命令行参数的个数不正确。"""


class FileReadError(PlagiarismError):
    """输入文件不存在、无法读取或内容为空。"""


class FileWriteError(PlagiarismError):
    """答案文件无法写入。"""


def read_text(path):
    """读取文本文件内容。

    依次尝试 UTF-8、GBK、GB18030 三种编码；全部失败时忽略非法字符强制
    解码，以保证程序不会因编码问题而中断。

    Args:
        path: 文件路径。

    Returns:
        文件的全部内容。

    Raises:
        FileReadError: 文件不存在、无法读取或内容为空时抛出。
    """
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
        # 三种编码都试过了仍然失败，忽略非法字符强制解码
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

    使用 jieba 精确模式，能较好识别「星期天」「电影」这类中文多字词。

    Args:
        text: 待分词的文本。

    Returns:
        过滤后的词列表。
    """
    words = jieba.lcut(text, cut_all=False)
    return [
        word
        for word in words
        if word.strip() and not all(char in NOISE_CHARS for char in word)
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
