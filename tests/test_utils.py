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
        "optim-optimizer-momentum": {0.9},
        "optim-learningrate": {0.4},
        "optim-weightdecay": {0.0005},
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": {256},
        "epochs": {300},
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    assert clean_values(data) == true_dct


def test_clean_values_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    assert type(data) == dict
    key_lst = [
        "optim-optimizer-Adam",
        "optim-optimizer-MomentumSGD",
        "optim-optimizer-momentum",
    ]
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-momentum": {0.9},
    }
    assert clean_values(data, key_lst) == true_dct


def test_clean_values_4():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    assert type(data) == dict
    key_lst = [
        "not_exist_0",
        "not_exist_1",
    ]
    true_dct = {}
    assert clean_values(data, key_lst) == true_dct


def test_concat_json_0():
    dct1 = {
        "epochs": 0,
        "optim-optimizer-Adam": True,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct2 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct3 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
        "iterations": 5510,
    }
    expected = {
        "epochs": {0, 1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
    }
    expected_majority_vote = {
        "epochs": {1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
    }
    assert concat_json_result([dct1, dct2, dct3]) == expected
    assert (
        concat_json_result([dct1, dct2, dct3], vote_option="majority_vote")
        == expected_majority_vote
    )


def test_concat_json_1():
    dct1 = {
        "epochs": 0,
        "optim-optimizer-Adam": True,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct2 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct3 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 0.00005,
        "iterations": 5510,
    }
    expected = {
        "epochs": {0, 1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
    }
    expected_majority_vote = {
        "epochs": {1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
    }

    assert concat_json_result([dct1, dct2, dct3]) == expected
    assert (
        concat_json_result([dct1, dct2, dct3], vote_option="majority_vote")
        == expected_majority_vote
    )


def test_concat_json_2():
    dct1 = {
        "epochs": 0,
        "optim-optimizer-Adam": True,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct2 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct3 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 0.00005,
        "iterations": 5510,
    }
    dct4 = {
        "epochs": 1,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 0.00005,
        "FPS": 1e5,
    }
    expected = {
        "epochs": {0, 1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
        "FPS": {100000},
    }
    expected_majority_vote = {
        "epochs": {1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510},
        "FPS": {100000},
    }
    assert concat_json_result([dct1, dct2, dct3, dct4]) == expected
    assert (
        concat_json_result([dct1, dct2, dct3, dct4], vote_option="majority_vote")
        == expected_majority_vote
    )


def test_concat_json_3():
    dct1 = {
        "epochs": 0,
        "optim-optimizer-Adam": True,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct2 = {
        "epochs": 0,
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-5,
    }
    dct3 = {
        "epochs": 1,
        "optim-weightdecay": 4e-5,
        "iterations": 5510,
    }
    dct4 = {
        "epochs": 1,
        "iterations": 100,
    }
    expected = {
        "epochs": {0, 1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5, 4e-5},
        "iterations": {5510, 100},
    }
    expected_majority_vote = {
        "epochs": {0, 1},
        "optim-optimizer-Adam": True,
        "optim-weightdecay": {5e-5},
        "iterations": {5510, 100},
    }
    assert concat_json_result([dct1, dct2, dct3, dct4]) == expected
    assert (
        concat_json_result([dct1, dct2, dct3, dct4], vote_option="majority_vote")
        == expected_majority_vote
    )
