"""文件读取模块。

负责把磁盘上的文本安全地读入内存：依次尝试多种常见中文编码，
全部失败时忽略非法字符强制解码，避免程序因编码问题而中断；
文件不存在、不可读或内容为空时统一抛出 FileReadError。
"""

from errors import FileReadError

# 按优先级依次尝试的编码。
# 首选用 utf-8-sig 而非 utf-8：前者能自动剥离文件开头的 BOM
# （Windows 记事本保存 UTF-8 时会写入 BOM），对不带 BOM 的文件同样有效。
ENCODINGS = ("utf-8-sig", "gbk", "gb18030")

# 字节顺序标记，个别环境下可能残留，读取后统一清除
BOM = "\ufeff"


def read_text(path):
    """读取文本文件内容。

    Args:
        path: 文件路径。

    Returns:
        文件的全部内容。

    Raises:
        FileReadError: 文件不存在、无法读取或内容为空时抛出。
    """
    text = None
    for encoding in ENCODINGS:
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

    # 清除可能残留的 BOM，避免它被当成一个词参与相似度统计
    text = text.lstrip(BOM)

    if not text.strip():
        raise FileReadError(f"文件内容为空：{path}")

    return text
