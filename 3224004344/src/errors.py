"""本程序的自定义异常体系。

所有异常都继承自 PlagiarismError，入口处只需捕获这一个基类，
即可统一处理参数、读取与写入三个环节的各类错误，
保证程序在任何异常输入下都不会抛出堆栈而崩溃。
"""


class PlagiarismError(Exception):
    """本程序所有自定义异常的基类，便于在入口处统一捕获。"""


class ParameterError(PlagiarismError):
    """命令行参数的个数不正确。"""


class FileReadError(PlagiarismError):
    """输入文件不存在、无法读取或内容为空。"""


class FileWriteError(PlagiarismError):
    """答案文件无法写入。"""
