import sys

import pytest

sys.path.append("../gptsenpy")


from gptsenpy.io.read import read_json, read_text


def test_read_json():
    data_path = "tests/data/hand_replacer.json"
    data = read_json(data_path)
    assert type(data) == dict
    expected = {"name": "John", "age": "30"}
    assert data == expected


def test_read_text():
    data_path = "tests/data/hand_prompt.txt"
    data = read_text(data_path)
    assert type(data) == str
    expected = "Hello, my name is ${name} and I am ${age} years old."

    assert data == expected


def test_duplicate_keys():
    data_path = "tests/data/duplicate_keys.json"
    with pytest.raises(AssertionError) as e:
        _ = read_json(data_path)
    assert str(e.value) == "Duplicate keys in JSON file."
