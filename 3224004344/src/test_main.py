"""论文查重程序的单元测试。

覆盖全部 6 个模块（errors / file_reader / text_processor / similarity /
answer_writer / main），共 39 个用例，按「等价类划分 + 边界值 + 异常路径」
三类方法设计，白盒覆盖每个函数的正常分支与异常分支；其中用例 36~38
是依据覆盖率报告的 Missing 列补充的定向用例，用例 39 为交付前
端到端验收发现的 UTF-8 BOM 缺陷所补的回归用例。

运行方式（在 3224004344 目录下执行）：
    py -3.13 -m unittest test_main -v
    py -3.13 -m coverage run -m unittest test_main
"""

import io
import os
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import main
from answer_writer import write_answer
from errors import (
    FileReadError,
    FileWriteError,
    ParameterError,
    PlagiarismError,
)
from file_reader import read_text
from similarity import calc_similarity, cosine_similarity
from text_processor import build_word_freq, tokenize

# 题面给出的样例文本，作为端到端用例的固定输入
SAMPLE_ORIGINAL = "今天是星期天，天气晴，今天晚上我要去看电影。"
SAMPLE_COPIED = "今天是周天，天气晴朗，我晚上要去看电影。"
SAMPLE_SIMILARITY = "0.61"

# 入口脚本的绝对路径，用于脚本方式运行的用例
MAIN_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "main.py"
)


class TempDirTestCase(unittest.TestCase):
    """为需要磁盘文件的用例提供统一的临时目录环境。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def make_file(self, name, content, encoding="utf-8"):
        """在临时目录中创建文本文件，返回其路径。"""
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding=encoding) as handle:
            handle.write(content)
        return path

    def make_bytes(self, name, content):
        """在临时目录中写入原始字节，用于构造编码异常场景。"""
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(content)
        return path


class TestExceptionHierarchy(unittest.TestCase):
    """用例 1-2：异常体系（errors.py）。"""

    def test_01_subclasses_inherit_base(self):
        """三个子类都必须继承基类，入口才能一处捕获。"""
        for cls in (ParameterError, FileReadError, FileWriteError):
            self.assertTrue(issubclass(cls, PlagiarismError))

    def test_02_base_class_catches_subclass(self):
        """捕获基类 PlagiarismError 应能拦截子类异常。"""
        with self.assertRaises(PlagiarismError):
            raise FileWriteError("模拟写入失败")


class TestFileReader(TempDirTestCase):
    """用例 3-9：文件读取（file_reader.py）。"""

    def test_03_read_utf8(self):
        """等价类：UTF-8 编码的中文文件应被正确读取。"""
        path = self.make_file("utf8.txt", "软件工程")
        self.assertEqual(read_text(path), "软件工程")

    def test_04_read_gbk(self):
        """等价类：GBK 编码文件应回退解码成功。"""
        path = self.make_file("gbk.txt", "论文查重", encoding="gbk")
        self.assertEqual(read_text(path), "论文查重")

    def test_05_missing_file_raises(self):
        """异常路径：文件不存在时抛出 FileReadError。"""
        path = os.path.join(self.dir, "not_exist.txt")
        with self.assertRaises(FileReadError):
            read_text(path)

    def test_06_directory_raises(self):
        """异常路径：路径指向目录时抛出 FileReadError。"""
        with self.assertRaises(FileReadError):
            read_text(self.dir)

    def test_07_empty_file_raises(self):
        """边界值：空文件被视为无效输入。"""
        path = self.make_file("empty.txt", "")
        with self.assertRaises(FileReadError):
            read_text(path)

    def test_08_blank_file_raises(self):
        """边界值：仅含空白字符的文件同样被视为空。"""
        path = self.make_file("blank.txt", "   \n\t  ")
        with self.assertRaises(FileReadError):
            read_text(path)

    def test_09_invalid_bytes_fallback(self):
        """异常路径：三种编码均失败时，忽略非法字符兜底解码。"""
        path = self.make_bytes("bad.txt", b"hello\x80world")
        self.assertEqual(read_text(path), "helloworld")

    def test_36_fallback_open_error_raises(self):
        """覆盖率补测：兜底解码阶段仍打不开文件时抛出 FileReadError。"""
        path = self.make_bytes("bad.txt", b"\x80")
        real_open = open
        state = {"calls": 0}

        def fake_open(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] > 3:
                raise PermissionError("模拟无法打开文件")
            return real_open(*args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            with self.assertRaises(FileReadError):
                read_text(path)

    def test_39_utf8_bom_is_stripped(self):
        """边界值：带 BOM 的 UTF-8 文件（Windows 记事本保存的默认格式）
        读取后不应残留 BOM 字符，否则它会被当作一个词参与相似度统计。
        """
        path = self.make_file("bom.txt", "软件工程", encoding="utf-8-sig")
        text = read_text(path)
        self.assertEqual(text, "软件工程")
        self.assertNotIn("\ufeff", text)


class TestTextProcessor(unittest.TestCase):
    """用例 10-14：文本处理（text_processor.py）。"""

    def test_10_tokenize_filters_punctuation(self):
        """分词结果中不应残留任何标点符号。"""
        words = list(tokenize("今天，天气晴。"))
        self.assertNotIn("，", words)
        self.assertNotIn("。", words)
        self.assertIn("天气", words)

    def test_11_tokenize_returns_generator(self):
        """返回值应为生成器，支持边分词边统计以降低内存峰值。"""
        self.assertIsInstance(tokenize("测试文本"), types.GeneratorType)

    def test_12_tokenize_punctuation_only_is_empty(self):
        """边界值：纯标点文本分词后没有任何有效词。"""
        self.assertEqual(list(tokenize("，。！？；：")), [])

    def test_13_build_word_freq_counts(self):
        """词频统计应正确累计每个词的出现次数。"""
        freq = build_word_freq(["软件", "工程", "软件"])
        self.assertEqual(freq["软件"], 2)
        self.assertEqual(freq["工程"], 1)

    def test_14_build_word_freq_empty(self):
        """边界值：空输入的词频字典长度为 0。"""
        self.assertEqual(len(build_word_freq([])), 0)


class TestSimilarity(TempDirTestCase):
    """用例 15-23：相似度计算（similarity.py）。"""

    def test_15_identical_text_is_one(self):
        """完全相同文本的相似度为 1。"""
        freq = build_word_freq(tokenize(SAMPLE_ORIGINAL))
        self.assertAlmostEqual(
            cosine_similarity(freq, freq), 1.0, places=9
        )

    def test_16_unrelated_text_is_zero(self):
        """完全无关的两段文本相似度为 0。"""
        freq_a = build_word_freq(tokenize("软件工程与项目管理"))
        freq_b = build_word_freq(tokenize("今天去公园散步买水果"))
        self.assertAlmostEqual(
            cosine_similarity(freq_a, freq_b), 0.0, places=9
        )

    def test_17_one_side_empty_is_zero(self):
        """边界值：一方词频为空时返回 0，且不触发除零。"""
        freq = build_word_freq(tokenize(SAMPLE_ORIGINAL))
        self.assertEqual(cosine_similarity({}, freq), 0.0)

    def test_18_both_empty_is_zero(self):
        """边界值：双方词频都为空时返回 0。"""
        self.assertEqual(cosine_similarity({}, {}), 0.0)

    def test_19_symmetry(self):
        """余弦相似度满足对称性：cos(a, b) == cos(b, a)。"""
        freq_a = build_word_freq(tokenize(SAMPLE_ORIGINAL))
        freq_b = build_word_freq(tokenize(SAMPLE_COPIED))
        self.assertAlmostEqual(
            cosine_similarity(freq_a, freq_b),
            cosine_similarity(freq_b, freq_a),
            places=9,
        )

    def test_20_range_within_zero_and_one(self):
        """结果必须落在 [0, 1] 区间内。"""
        freq_a = build_word_freq(tokenize(SAMPLE_ORIGINAL))
        freq_b = build_word_freq(tokenize(SAMPLE_COPIED))
        value = cosine_similarity(freq_a, freq_b)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_21_scale_invariance(self):
        """词频整体放大不影响相似度（分母抵消了文本长度）。"""
        freq_a = build_word_freq(tokenize(SAMPLE_ORIGINAL))
        freq_b = build_word_freq(tokenize(SAMPLE_COPIED))
        doubled = {word: count * 2 for word, count in freq_a.items()}
        self.assertAlmostEqual(
            cosine_similarity(doubled, freq_b),
            cosine_similarity(freq_a, freq_b),
            places=9,
        )

    def test_22_calc_similarity_end_to_end(self):
        """集成路径：由文件路径直接算出题面样例的重复率。"""
        original = self.make_file("orig.txt", SAMPLE_ORIGINAL)
        copied = self.make_file("copy.txt", SAMPLE_COPIED)
        value = calc_similarity(original, copied)
        self.assertEqual(f"{value:.2f}", SAMPLE_SIMILARITY)

    def test_23_calc_similarity_missing_file(self):
        """异常路径：任一输入文件缺失时向上抛出 FileReadError。"""
        original = self.make_file("orig.txt", SAMPLE_ORIGINAL)
        missing = os.path.join(self.dir, "not_exist.txt")
        with self.assertRaises(FileReadError):
            calc_similarity(original, missing)

    def test_37_zero_vector_is_zero(self):
        """覆盖率补测：向量模长为 0 时直接返回 0，不触发除零。"""
        self.assertEqual(cosine_similarity({"词": 0}, {"词": 1}), 0.0)


class TestAnswerWriter(TempDirTestCase):
    """用例 24-27：答案写出（answer_writer.py）。"""

    def test_24_write_two_decimals(self):
        """输出应四舍五入并保留两位小数。"""
        path = os.path.join(self.dir, "ans.txt")
        text = write_answer(path, 0.6149)
        self.assertEqual(text, "0.61")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "0.61")

    def test_25_write_boundary_values(self):
        """边界值：0 与 1 都应格式化为两位小数。"""
        path = os.path.join(self.dir, "ans.txt")
        self.assertEqual(write_answer(path, 0.0), "0.00")
        self.assertEqual(write_answer(path, 1.0), "1.00")

    def test_26_overwrite_truncates(self):
        """覆盖写入时应截断旧内容，不残留上一次的结果。"""
        path = self.make_file("ans.txt", "0.123456789")
        write_answer(path, 0.5)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "0.50")

    def test_27_unwritable_path_raises(self):
        """异常路径：目标目录不存在时抛出 FileWriteError。"""
        path = os.path.join(self.dir, "no_such_dir", "ans.txt")
        with self.assertRaises(FileWriteError):
            write_answer(path, 0.5)


class TestParseArgs(unittest.TestCase):
    """用例 28-31：参数解析（main.parse_args）。"""

    def test_28_parse_three_args(self):
        """等价类：正确传入 3 个参数时返回三元组。"""
        argv = ["main.py", "a.txt", "b.txt", "c.txt"]
        self.assertEqual(main.parse_args(argv), ("a.txt", "b.txt", "c.txt"))

    def test_29_zero_args_raises(self):
        """异常路径：参数个数为 0 时抛出 ParameterError。"""
        with self.assertRaises(ParameterError):
            main.parse_args(["main.py"])

    def test_30_two_args_raises(self):
        """异常路径：参数个数为 2 时抛出 ParameterError。"""
        with self.assertRaises(ParameterError):
            main.parse_args(["main.py", "a.txt", "b.txt"])

    def test_31_four_args_raises(self):
        """边界值：多传参数同样视为参数个数错误。"""
        argv = ["main.py", "a.txt", "b.txt", "c.txt", "d.txt"]
        with self.assertRaises(ParameterError):
            main.parse_args(argv)


class TestMain(TempDirTestCase):
    """用例 32-35：程序入口（main.main）端到端。"""

    def _run(self, argv):
        """在静默标准输出的情况下调用 main()，返回退出码。"""
        with mock.patch.object(sys, "argv", argv):
            with redirect_stdout(io.StringIO()):
                return main.main()

    def test_32_success_returns_zero(self):
        """正常路径：退出码 0，答案文件内容为题面样例的重复率。"""
        original = self.make_file("orig.txt", SAMPLE_ORIGINAL)
        copied = self.make_file("copy.txt", SAMPLE_COPIED)
        answer = os.path.join(self.dir, "ans.txt")
        code = self._run(["main.py", original, copied, answer])
        self.assertEqual(code, 0)
        with open(answer, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), SAMPLE_SIMILARITY)

    def test_33_wrong_arg_count_returns_one(self):
        """异常路径：参数个数错误时退出码为 1（不崩溃）。"""
        self.assertEqual(self._run(["main.py"]), 1)

    def test_34_read_error_returns_one(self):
        """异常路径：输入文件缺失时退出码为 1（不崩溃）。"""
        missing = os.path.join(self.dir, "not_exist.txt")
        copied = self.make_file("copy.txt", SAMPLE_COPIED)
        answer = os.path.join(self.dir, "ans.txt")
        self.assertEqual(self._run(["main.py", missing, copied, answer]), 1)

    def test_35_write_error_returns_one(self):
        """异常路径：答案路径不可写时退出码为 1（不崩溃）。"""
        original = self.make_file("orig.txt", SAMPLE_ORIGINAL)
        copied = self.make_file("copy.txt", SAMPLE_COPIED)
        answer = os.path.join(self.dir, "no_such_dir", "ans.txt")
        self.assertEqual(self._run(["main.py", original, copied, answer]), 1)

    def test_38_script_exit_code(self):
        """覆盖率补测：以脚本方式运行时经 sys.exit 返回退出码 0。"""
        original = self.make_file("orig.txt", SAMPLE_ORIGINAL)
        copied = self.make_file("copy.txt", SAMPLE_COPIED)
        answer = os.path.join(self.dir, "ans.txt")
        argv = ["main.py", original, copied, answer]
        with mock.patch.object(sys, "argv", argv):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runpy.run_path(MAIN_SCRIPT, run_name="__main__")
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
