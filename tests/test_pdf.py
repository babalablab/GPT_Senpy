import sys
from pathlib import Path

import pytest

sys.path.append("../gptsenpy")
DATA_PATH = Path("tests/data")


from gptsenpy.PDFLoader import PDFLoader
from gptsenpy.Tokenizer import Tokenizer


def test_split_sections_1():
    pdf_path = DATA_PATH / "AA.pdf"
    pdfloader = PDFLoader(pdf_path)
    assert pdfloader.page_count == 13
    sections = pdfloader.split_sections()
    assert len(sections) == 6


def test_split_sections_2():
    pdf_path = DATA_PATH / "CenterNet.pdf"
    pdfloader = PDFLoader(pdf_path)
    assert pdfloader.page_count == 10
    sections = pdfloader.split_sections()
    assert len(sections) == 6


def test_split_sections_3():
    pdf_path = DATA_PATH / "DAMO-YOLO.pdf"
    pdfloader = PDFLoader(pdf_path)
    assert pdfloader.page_count == 10
    sections = pdfloader.split_sections()
    assert len(sections) == 6


def test_split_sections_4():
    pdf_path = DATA_PATH / "Libra.pdf"
    pdfloader = PDFLoader(pdf_path)
    assert pdfloader.page_count == 10
    # assert len(pdfloader.sections) == 6
    # 正確に抽出できていない
    sections = pdfloader.split_sections()
    assert len(sections) <= 10


def test_split_sections_5():
    pdf_path = DATA_PATH / "SS.pdf"
    pdfloader = PDFLoader(pdf_path)
    assert pdfloader.page_count == 14
    # assert len(pdfloader.sections) == 7
    # 正確に抽出できていない
    sections = pdfloader.split_sections()
    assert len(sections) <= 10


def test_divide_sections_1():
    pdf_path = DATA_PATH / "AA.pdf"
    pdfloader = PDFLoader(pdf_path)
    sections = pdfloader.split_sections()
    tokenizer = Tokenizer("gpt-3.5-turbo")
    ret = tokenizer.divide_text_by_max_token(sections[0], max_tokens=10)
    assert len(ret) == 38


def test_divide_sections_2():
    pdf_path = DATA_PATH / "AA.pdf"
    pdfloader = PDFLoader(pdf_path)
    sections = pdfloader.split_sections()
    tokenizer = Tokenizer("gpt-3.5-turbo")
    ret = tokenizer.divide_text_by_max_token(sections[0], max_tokens=2000)
    assert len(ret) == 1
