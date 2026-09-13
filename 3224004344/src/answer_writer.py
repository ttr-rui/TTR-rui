"""答案文件写出模块。

与 file_reader 模块职责对称：读取环节由 file_reader 负责，
写出环节由本模块负责。任何写入失败都统一转换为 FileWriteError，
使入口层只需捕获 PlagiarismError 一个基类。
"""

from errors import FileWriteError


def write_answer(answer_path, similarity):
    """把重复率写入答案文件，保留两位小数。

    Args:
        answer_path: 答案文件路径。
        similarity: 0~1 之间的重复率。

    Returns:
        实际写入文件的文本内容，形如 "0.61"。

    Raises:
        FileWriteError: 目标路径不可写（目录不存在、无权限等）时抛出。
    """
    text = f"{similarity:.2f}"
    try:
        with open(answer_path, "w", encoding="utf-8") as file:
            file.write(text)
    except OSError as error:
        raise FileWriteError(
            f"无法写入答案文件：{answer_path}（{error}）"
        ) from error
    return text
