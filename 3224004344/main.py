"""论文查重程序入口。

从命令行接收三个绝对路径参数（原文文件、抄袭版文件、答案文件），
计算两篇文章的重复率并写入答案文件。
"""

import sys


def main():
    """读取命令行参数，计算重复率并写出结果。"""
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件] [抄袭版文件] [答案文件]")
        sys.exit(1)

    original_path = sys.argv[1]
    copy_path = sys.argv[2]
    answer_path = sys.argv[3]

    # TODO: 第4步在此实现查重算法
    similarity = 0.0

    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(f"{similarity:.2f}")

    print(f"重复率已写入 {answer_path}")


if __name__ == "__&#8203;main__":
    main()
