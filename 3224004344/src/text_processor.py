"""文本处理模块。

负责把原始文本转换成词频向量：先统一替换标点与空白，
再用 jieba 精确模式分词，最后统计每个词的出现次数。
"""

from collections import Counter

import jieba

# 关闭 jieba 加载词典时的日志输出，保持终端干净（60 高于 CRITICAL）
jieba.setLogLevel(60)

# 标点符号与空白字符不携带语义，分词前统一替换为空格
NOISE_CHARS = set(
    "，。、；：？！“”‘’（）《》〈〉【】〔〕—…·「」『』"
    ",.;:?!\"'()[]{}<>|/\\-_=+*&^%$#@~` \t\r\n\v\f\u3000"
)

# 预构建翻译表：str.translate 在 C 层一次性完成全部替换，
# 避免在分词后对每个 token 逐字符判断是否属于标点
PUNCT_TABLE = str.maketrans({char: " " for char in NOISE_CHARS})


def tokenize(text):
    """把文本切分成词，返回有效词的生成器。

    分词前先用 str.translate 在 C 层把标点统一替换成空格，因此分词
    结果里不再混入标点，过滤条件简化为一次 strip()；以生成器形式
    返回，调用方可以边分词边统计，无需在内存中落地完整词列表。

    Args:
        text: 待分词的文本。

    Returns:
        过滤掉标点与空白之后的词生成器。
    """
    cleaned = text.translate(PUNCT_TABLE)
    return (
        word for word in jieba.cut(cleaned, cut_all=False) if word.strip()
    )


def build_word_freq(tokens):
    """统计词频，构造「词 → 出现次数」的映射。

    直接消费词迭代器，不产生中间列表。

    Args:
        tokens: 词的可迭代对象。

    Returns:
        词频计数（Counter，dict 的子类）；输入为空时返回空 Counter。
    """
    return Counter(tokens)
