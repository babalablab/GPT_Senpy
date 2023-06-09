import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import Metrics
from gptsenpy.utils import clean_values


def test_precision_recall_0():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    mt = Metrics(data, data)
    assert mt.precision == 0.0 and mt.recall == 0.0


def test_precision_recall_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, data)
    assert mt.precision == 1.0 and mt.recall == 1.0


def test_precision_recall_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.precision == 0.0 and mt.recall == 0.0


def test_precision_recall_3():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.precision == 0.0 and mt.recall == 0.0


def test_precision_recall_4():
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
    assert mt.precision == 0.0 and mt.recall == 0.0


def test_precision_recall_7():
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
    assert mt.precision == 0.0 and mt.recall == 0.0


def test_precision_recall_5():
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
    assert mt.precision == 7 / 8 and mt.recall == 7 / 8


def test_precision_recall_6():
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
    assert mt.precision == 5 / 6
    assert mt.recall == 5 / 8


def test_precision_recall_9():
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
    assert mt.get_precision(data, preds) == 5 / 6
    assert mt.get_recall(data, preds) == 5 / 8
    assert mt.precision == 5 / 6
    assert mt.recall == 5 / 8


def test_precision_recall_8():
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
    assert mt.get_precision(data, {}) == 0
    assert mt.get_recall({}, preds) == 0
    assert mt.precision == 5 / 6
    assert mt.recall == 5 / 8


def test_precision_recall_10():
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
    assert mt.get_precision(data, data) == 1.0
    assert mt.get_recall(preds, preds) == 1.0
    assert mt.precision == 5 / 6
    assert mt.recall == 5 / 8


def test_precision_recall_11():
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
    assert mt.precision == 4 / 5
    assert mt.recall == 4 / 8


def test_precision_recall_12():
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
    assert mt.precision == 5 / 7
    assert mt.recall == 5 / 8


def test_precision_recall_13():
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
    assert mt.precision == 7 / 9
    assert mt.recall == 7 / 8


def test_precision_recall_14():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [0.9, 0.9],
        "optim-learningrate": [4e-01],
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 7 / 8
    assert mt.recall == 7 / 8


def test_precision_recall_15():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [9000e-4, 0.9],
        "optim-learningrate": [4e-01],
        "optim-weightdecay": True,
        "optim-lrschedular": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 7 / 8
    assert mt.recall == 7 / 8
