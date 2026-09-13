"""论文查重程序入口。

从命令行接收三个绝对路径参数（原文文件、抄袭版论文文件、答案文件），
调用相似度计算模块得到重复率，并将结果写入答案文件。

本文件只负责「解析参数 → 调度 → 写结果」三件事，
具体的文件读取、文本处理与相似度计算分别由
file_reader.py、text_processor.py、similarity.py 承担。

用法：
    python main.py [原文文件] [抄袭版论文的文件] [答案文件]
"""

import sys

from errors import PlagiarismError
from similarity import calc_similarity


def main():
    """解析命令行参数、调度查重逻辑、写出结果。

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
