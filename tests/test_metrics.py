import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import Metrics
from gptsenpy.metrics.metrics import clean_values


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


def test_recall_precision_0():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    mt = Metrics(data, data)
    assert mt.recall == 0.0 and mt.precision == 0.0


def test_recall_precision_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, data)
    assert mt.recall == 1.0 and mt.precision == 1.0


def test_recall_precision_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.recall == 0.0 and mt.precision == 0.0


def test_recall_precision_3():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.recall == 0.0 and mt.precision == 0.0


def test_recall_precision_4():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 300,
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, true_dct)
    assert mt.recall == 0.0 and mt.precision == 0.0


def test_recall_precision_7():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 300,
        "resource-gpu-T4": True,
    }
    mt = Metrics(true_dct, data)
    assert mt.recall == 0.0 and mt.precision == 0.0


def test_recall_precision_5():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.recall == 7 / 8 and mt.precision == 7 / 8


def test_recall_precision_6():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.recall == 5 / 6
    assert mt.precision == 5 / 8


def test_recall_precision_9():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_recall(data, preds) == 5 / 6
    assert mt.get_precision(data, preds) == 5 / 8
    assert mt.recall == 5 / 6
    assert mt.precision == 5 / 8


def test_recall_precision_8():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_recall(data, {}) == 0
    assert mt.get_precision({}, preds) == 0
    assert mt.recall == 5 / 6
    assert mt.precision == 5 / 8


def test_recall_precision_10():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_recall(data, data) == 1.0
    assert mt.get_precision(preds, preds) == 1.0
    assert mt.recall == 5 / 6
    assert mt.precision == 5 / 8


def test_recall_precision_11():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": True,
        "optim-lrschedular": False,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.recall == 4 / 5
    assert mt.precision == 4 / 8


def test_recall_precision_12():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": [0.4, 0.7],
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.recall == 5 / 7
    assert mt.precision == 5 / 8


def test_recall_precision_13():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [0.9, 0.9],
        "optim-learningrate": [0.4, 0.9],
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.recall == 7 / 9
    assert mt.precision == 7 / 8
