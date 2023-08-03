import json
import sys

sys.path.append("../gptsenpy")
from gptsenpy.io.read import read_json
from gptsenpy.utils import clean_values, categorize_dict_keys, uncategorize_dict_keys


with open("tests/data/label_category.json", "r") as f:
    label_category = json.load(f)


def test_clean_values_null():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    assert type(data) == dict
    assert clean_values(data) == {}


def test_clean_values_1():
    data_path = "tests/data/Attention_Augmented_Convolutional_Networks.json"
    data = read_json(data_path)
    assert type(data) == dict
    assert clean_values(data) == {"epochs": 150}


def test_clean_values_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    assert type(data) == dict
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 300,
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
        "optim-optimizer-momentum": 0.9,
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


def test_categorize_dict_keys_0():
    dct1 = {
        "optim-optimizer-Adam": True,
        "optim-lrscheduler-LambdaLR": True,
        "optim-weightdecay": 5e-5,
        "epochs": 10,
    }
    dct2 = {
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "optim-weightdecay": 0.00005,
        "epochs": 20,
    }
    dct3 = {
        "optim-optimizer-SGD": True,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "optim-weightdecay": 5e-4,
        "epochs": 20,
        "iterations": 5510,
    }
    label_category = {
        "optim-optimizer": ["optim-optimizer-Adam", "optim-optimizer-SGD"],
        "optim-lrscheduler": [
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        ],
        "optim-weightdecay": ["optim-weightdecay"],
        "epochs": ["epochs"],
        "iterations": ["iterations"],
    }
    expected = {
        "optim-optimizer": {"optim-optimizer-Adam", "optim-optimizer-SGD"},
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "optim-weightdecay": {5e-5, 5e-4},
        "epochs": {10, 20},
        "iterations": {5510},
    }
    expected_majority_vote = {
        "optim-optimizer": {"optim-optimizer-Adam", "optim-optimizer-SGD"},
        "optim-lrscheduler": {"optim-lrscheduler-CosineAnnealingLR"},
        "optim-weightdecay": {5e-5},
        "epochs": {20},
        "iterations": {5510},
    }
    assert categorize_dict_keys([dct1, dct2, dct3], label_category) == expected
    assert (
        categorize_dict_keys([dct1, dct2, dct3], label_category, "majority_vote")
        == expected_majority_vote
    )


def test_categorize_dict_keys_1():
    dct1 = {
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-LambdaLR": True,
        "optim-weightdecay": 5e-5,
        "optim-earlystopping": True,
        "epochs": 10,
    }
    dct2 = {
        "optim-optimizer-Adam": False,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "optim-weightdecay": 5e-4,
        "optim-earlystopping": False,
        "epochs": 20,
    }
    dct3 = {
        "optim-optimizer-not_exist": True,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "optim-weightdecay": 5e-3,
    }
    label_category = {
        "optim-optimizer": ["optim-optimizer-Adam", "optim-optimizer-SGD"],
        "optim-lrscheduler": [
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        ],
        "optim-weightdecay": ["optim-weightdecay"],
        "optim-earlystopping": ["optim-earlystopping"],
        "epochs": ["epochs"],
        "iterations": ["iterations"],  # not include in any of dct1, dct2, and dct3
    }
    expected = {
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "optim-weightdecay": {5e-5, 5e-4, 5e-3},
        "optim-earlystopping": {True},
        "epochs": {10, 20},
    }
    expected_majority_vote = {
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "optim-weightdecay": {5e-5, 5e-4, 5e-3},
        "optim-earlystopping": {True},
        "epochs": {10, 20},
    }
    assert categorize_dict_keys([dct1, dct2, dct3], label_category) == expected
    assert (
        categorize_dict_keys([dct1, dct2, dct3], label_category, "majority_vote")
        == expected_majority_vote
    )


def test_categorize_dict_keys_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), label_category)  # one dict
    expected = {
        "optim-optimizer": {"optim-optimizer-MomentumSGD"},
        "optim-optimizer-momentum": {0.9},
        "optim-learningrate": {0.4},
        "optim-weightdecay": {0.0005},
        "optim-lrscheduler": {"optim-lrscheduler-CosineAnnealingLR"},
        "batchsize": {256},
        "epochs": {300},
        "resource-train-gpu": {"resource-train-gpu-T4"},
        "resource-inference-gpu": {"resource-inference-gpu-T4"},
    }
    assert data == expected


def test_categorize_dict_keys_3():
    data_path = "tests/data/annotation_format.json"
    data = categorize_dict_keys(read_json(data_path), label_category)  # empty dict
    assert data == {}


def test_categorize_dict_keys_4():
    try:
        categorize_dict_keys([{}], {}, "not_exist")
    except ValueError:
        pass
    else:
        assert 1


def test_uncategorize_dict_keys_0():
    dct = {
        "optim-optimizer": {"optim-optimizer-Adam", "optim-optimizer-SGD"},
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "optim-weightdecay": {5e-5, 5e-4},
        "epochs": {10, 20},
        "iterations": {5510},
    }
    label_category = {
        "optim-optimizer": ["optim-optimizer-Adam", "optim-optimizer-SGD"],
        "optim-lrscheduler": [
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        ],
        "optim-weightdecay": ["optim-weightdecay"],
        "epochs": ["epochs"],
        "iterations": ["iterations"],
    }
    expected = {
        "optim-optimizer-Adam": True,
        "optim-optimizer-SGD": True,
        "optim-lrscheduler-LambdaLR": True,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "optim-weightdecay": [5e-5, 5e-4],
        "epochs": [10, 20],
        "iterations": [5510],
    }
    output = uncategorize_dict_keys(dct, label_category)
    for k, v in expected.items():
        assert k in output
        if isinstance(v, bool):
            assert isinstance(output[k], bool) and output[k] == v
        else:
            assert isinstance(output[k], list) and set(output[k]) == set(v)


def test_uncategorize_dict_keys_1():
    dct = {
        "optim-optimizer": set(),
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "optim-weightdecay": {5e-5, 5e-4, 5e-3},
        "optim-earlystopping": {True},
        "epochs": {10, 20},
    }
    label_category = {
        "optim-optimizer": ["optim-optimizer-Adam", "optim-optimizer-SGD"],
        "optim-lrscheduler": [
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        ],
        "optim-weightdecay": ["optim-weightdecay"],
        "optim-earlystopping": ["optim-earlystopping"],
        "epochs": ["epochs"],
        "iterations": ["iterations"],  # not include in dct
    }
    expected = {
        "optim-lrscheduler-LambdaLR": True,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "optim-weightdecay": [5e-5, 5e-4, 5e-3],
        "optim-earlystopping": True,
        "epochs": [10, 20],
    }
    output = uncategorize_dict_keys(dct, label_category)
    for k, v in expected.items():
        assert k in output
        if isinstance(v, bool):
            assert isinstance(output[k], bool) and output[k] == v
        else:
            assert isinstance(output[k], list) and set(output[k]) == set(v)
