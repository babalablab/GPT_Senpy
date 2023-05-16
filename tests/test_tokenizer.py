import sys

sys.path.append("../gptsenpy")

from pytest import raises, warns

from gptsenpy.Tokenizer import Tokenizer

model = "gpt-4"
text = "Hello, my name is John"
tokenizer = Tokenizer(model)


def test_tokenizer():
    tokens = tokenizer.tokenize(text)
    assert len(tokens) >= 0


def test_count_tokenizer():
    n = tokenizer.count_tokens(text)
    assert n >= 0


def test_error_count_tokenizer():
    with raises(ValueError):
        _ = tokenizer.count_tokens(10000000)
