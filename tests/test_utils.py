import sys

sys.path.append("../gptsenpy")
from gptsenpy.io.read import read_json
from gptsenpy.utils import clean_values, concat_json_result


def test_clean_values_null():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    assert type(data) == dict
    assert clean_values(data) == {}


def test_clean_values_1():
    data_path = "tests/data/Attention_Augmented_Convolutional_Networks.json"
    data = read_json(data_path)
    assert type(data) == dict
    assert clean_values(data) == {"epochs": {150}}


def test_clean_values_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    assert type(data) == dict
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": {0.9},
        "optim-learningrate": {0.4},
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": {256},
        "epochs": {300},
        "resource-gpu-T4": True,
    }
    assert clean_values(data) == true_dct


def test_clean_values_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    assert type(data) == dict
    key_lst = ["optim-optimizer-Adam",
               "optim-optimizer-MomentumSGD",
               "optim-optimizer-MomentumSGD-momentum",
               "not_exist_0",
               "not_exist_1"]
    true_dct = {"optim-optimizer-MomentumSGD": True,
                "optim-optimizer-MomentumSGD-momentum": {0.9}}
    assert clean_values(data, key_lst) == true_dct


def test_concat_json_0():
    dct1 = {"a": 0, "b": True, "c": False, "d": 5e-5}
    dct2 = {"a": 1, "b": False, "c": False, "d": 5e-5}
    dct3 = {"a": 1, "b": False, "c": False, "d": 5e-5, "e": 5510}
    expected = {"a": {0, 1}, "b": True, "d": {5e-5}, "e": {5510}}
    assert concat_json_result([dct1, dct2, dct3]) == expected


def test_concat_json_1():
    dct1 = {"a": 0, "b": True, "c": False, "d": 5e-5}
    dct2 = {"a": 1, "b": False, "c": False, "d": 5e-5}
    dct3 = {"a": 1, "b": False, "c": False, "d": 0.00005, "e": 5510}
    expected = {"a": {0, 1}, "b": True, "d": {5e-5}, "e": {5510}}
    assert concat_json_result([dct1, dct2, dct3]) == expected


def test_concat_json_2():
    dct1 = {"a": 0, "b": True, "c": False, "d": 5e-5}
    dct2 = {"a": 1, "b": False, "c": False, "d": 5e-5}
    dct3 = {"a": 1, "b": False, "c": False, "d": 0.00005, "e": 5510}
    dct4 = {"a": 1, "b": False, "c": False, "d": 0.00005, "f": 1e5}
    expected = {"a": {0, 1}, "b": True, "d": {5e-5}, "e": {5510}, "f": {100000}}
    assert concat_json_result([dct1, dct2, dct3, dct4]) == expected
