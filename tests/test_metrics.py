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
    score_dict = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, data)
    assert mt.precision == 1.0 and mt.recall == 1.0
    score_dict = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.precision == 0.0 and mt.recall == 0.0
    score_dict = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_3():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    mt = Metrics(data, {})
    assert mt.precision == 0.0 and mt.recall == 0.0
    score_dict = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_4():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0001,
        "optim-lrscheduler-LambdaLR": True,
        "batchsize": 256,
        "epochs": 300,
        "resource-train-gpu-T4": True,
    }
    mt = Metrics(data, true_dct)
    assert mt.precision == 0.0 and mt.recall == 0.0
    score_dict = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_7():
    data_path = "tests/data/annotation_format.json"
    data = read_json(data_path)
    true_dct = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0001,
        "optim-lrscheduler-LambdaLR": True,
        "batchsize": 256,
        "epochs": 300,
        "resource-train-gpu-T4": True,
    }
    mt = Metrics(true_dct, data)
    assert mt.precision == 0.0 and mt.recall == 0.0
    score_dict = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert mt.export_metrics() == score_dict


def test_precision_recall_5():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 8 / 9 and mt.recall == 8 / 9
    f1 = 2 * (8 / 9) * (8 / 9) / (8 / 9 + 8 / 9)
    score_dict = {"precision": 8 / 9, "recall": 8 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_6():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 6 / 7
    assert mt.recall == 6 / 9
    f1 = 2 * (6 / 7) * (6 / 9) / (6 / 7 + 6 / 9)
    score_dict = {"precision": 6 / 7, "recall": 6 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_9():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_precision(data, preds) == 6 / 7
    assert mt.get_recall(data, preds) == 6 / 9
    assert mt.precision == 6 / 7
    assert mt.recall == 6 / 9


def test_precision_recall_8():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_precision(data, {}) == 0
    assert mt.get_recall({}, preds) == 0
    assert mt.precision == 6 / 7
    assert mt.recall == 6 / 9


def test_precision_recall_10():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.get_precision(data, data) == 1.0
    assert mt.get_recall(preds, preds) == 1.0
    assert mt.precision == 6 / 7
    assert mt.recall == 6 / 9


def test_precision_recall_11():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": 0.4,
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": False,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 5 / 6
    assert mt.recall == 5 / 9
    f1 = 2 * (5 / 6) * (5 / 9) / (5 / 6 + 5 / 9)
    score_dict = {"precision": 5 / 6, "recall": 5 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_12():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        # "optim-optimizer-MomentumSGD": True,
        # "optim-optimizer-MomentumSGD-momentum": 0.9,
        "optim-learningrate": [0.4, 0.7],
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 6 / 8
    assert mt.recall == 6 / 9
    f1 = 2 * (6 / 8) * (6 / 9) / (6 / 8 + 6 / 9)
    score_dict = {"precision": 6 / 8, "recall": 6 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_13():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [0.9, 0.9],
        "optim-learningrate": [0.4, 0.9],
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 8 / 10
    assert mt.recall == 8 / 9
    f1 = 2 * (8 / 10) * (8 / 9) / (8 / 10 + 8 / 9)
    score_dict = {"precision": 8 / 10, "recall": 8 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_14():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [0.9, 0.9],
        "optim-learningrate": [4e-01],
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 8 / 9
    assert mt.recall == 8 / 9
    f1 = 2 * (8 / 9) * (8 / 9) / (8 / 9 + 8 / 9)
    score_dict = {"precision": 8 / 9, "recall": 8 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict


def test_precision_recall_15():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = read_json(data_path)
    preds = {
        "optim-optimizer-MomentumSGD": True,
        "optim-optimizer-MomentumSGD-momentum": [9000e-4, 0.9],
        "optim-learningrate": [4e-01],
        "optim-weightdecay": 0.0005,
        "optim-lrscheduler-CosineAnnealingLR": True,
        "batchsize": 256,
        "epochs": 200,  # 300 -> 200
        "resource-train-gpu-T4": True,
        "resource-inference-gpu-T4": True,
    }
    mt = Metrics(data, preds)
    assert mt.precision == 8 / 9
    assert mt.recall == 8 / 9
    f1 = 2 * (8 / 9) * (8 / 9) / (8 / 9 + 8 / 9)
    score_dict = {"precision": 8 / 9, "recall": 8 / 9, "f1": f1}
    assert mt.export_metrics() == score_dict
