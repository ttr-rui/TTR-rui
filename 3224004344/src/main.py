"""论文查重程序入口。

从命令行接收三个绝对路径参数（原文文件、抄袭版论文文件、答案文件），
调用相似度计算模块得到重复率，并将结果写入答案文件。

本文件只负责「解析参数 → 调度 → 写结果」三件事，
具体的文件读取、文本处理与相似度计算分别由
file_reader.py、text_processor.py、similarity.py、answer_writer.py承担。

用法：
    python main.py [原文文件] [抄袭版论文的文件] [答案文件]
"""

import sys

from answer_writer import write_answer
from errors import ParameterError, PlagiarismError
from similarity import calc_similarity

USAGE = "用法：python main.py [原文文件] [抄袭版论文的文件] [答案文件]"


def parse_args(argv):
    """校验并解析命令行参数。

    Args:
        argv: 命令行参数列表（含脚本名本身）。

    Returns:
        三元组 (原文路径, 抄袭版路径, 答案路径)。

    Raises:
        ParameterError: 参数个数不为 3 个时抛出。
    """
    if len(argv) != 4:
        raise ParameterError(
            "参数个数不正确：需要 3 个（原文、抄袭版、答案），"
            f"实际收到 {len(argv) - 1} 个。"
        )
    return argv[1], argv[2], argv[3]


def main():
    """解析命令行参数、调度查重逻辑、写出结果。

    Returns:
        进程退出码，0 表示成功，1 表示出错。
    """
    try:
        original_path, copied_path, answer_path = parse_args(sys.argv)
        similarity = calc_similarity(original_path, copied_path)
        write_answer(answer_path, similarity)
    except PlagiarismError as error:
        print(f"错误：{error}")
        print(USAGE)
        return 1

    print(f"重复率 {similarity:.2f} 已写入 {answer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
