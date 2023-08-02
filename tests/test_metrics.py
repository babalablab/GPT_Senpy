import json
import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import Metrics
from gptsenpy.utils import categorize_dict_keys


with open("tests/data/label_category.json", "r") as f:
    LABEL_CATEGORY = json.load(f)


def test_Metrics_0():
    data_path = "tests/data/annotation_format.json"
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)
    assert data == {}

    recalls, precisions = [], []
    num_labels, num_preds = [], []
    for _ in range(len(LABEL_CATEGORY)):
        mt = Metrics(set(), set())
        recalls.append(mt.recall)
        precisions.append(mt.precision)
        num_labels.append(mt.num_labels)
        num_preds.append(mt.num_preds)
    assert recalls == precisions == num_labels == num_preds == [0] * len(LABEL_CATEGORY)


def test_Metrics_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)
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

    recalls, precisions = [], []
    num_labels, num_preds = [], []
    for i in LABEL_CATEGORY:
        gt = data[i] if i in data else set()
        pred = data[i] if i in data else set()
        mt = Metrics(gt, pred)
        recalls.append(mt.recall)
        precisions.append(mt.precision)
        num_labels.append(mt.num_labels)
        num_preds.append(mt.num_preds)
    assert (
        num_labels
        == num_preds
        == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    )
    assert (
        recalls == precisions == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    )


def test_Metrics_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    recalls, precisions = [], []
    num_labels, num_preds = [], []
    for i in LABEL_CATEGORY:
        gt = data[i] if i in data else set()
        mt = Metrics(gt, set())
        recalls.append(mt.recall)
        precisions.append(mt.precision)
        num_labels.append(mt.num_labels)
        num_preds.append(mt.num_preds)
    assert recalls == precisions == num_preds == [0] * len(LABEL_CATEGORY)
    assert num_labels == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]


def test_Metrics_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    recalls, precisions = [], []
    num_labels, num_preds = [], []
    for i in LABEL_CATEGORY:
        pred = data[i] if i in data else set()
        mt = Metrics(set(), pred)
        recalls.append(mt.recall)
        precisions.append(mt.precision)
        num_labels.append(mt.num_labels)
        num_preds.append(mt.num_preds)
    assert recalls == precisions == num_labels == [0] * len(LABEL_CATEGORY)
    assert num_preds == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]


def test_Metrics_4():
    gts = {
        "optim-optimizer": {"optim-optimizer-Adam"},
        "optim-lrscheduler": {"optim-lrscheduler-CosineAnnealingLR"},
        "optim-earlystopping": {True},
        "batchsize": {16},
        "iterations": {5510},
        "epochs": {10},
    }
    preds = {
        "optim-optimizer": {"optim-optimizer-Adam", "optim-optimizer-SGD"},
        "optim-weightdecay": {5e-5, 5e-4},
        "optim-lrscheduler": {
            "optim-lrscheduler-LambdaLR",
            "optim-lrscheduler-CosineAnnealingLR",
        },
        "iterations": {5510},
        "epochs": {10, 20, 30},
    }

    recalls, precisions = [], []
    num_labels, num_preds = [], []
    for i in LABEL_CATEGORY:
        gt = gts[i] if i in gts else set()
        pred = preds[i] if i in preds else set()
        mt = Metrics(gt, pred)
        recalls.append(mt.recall)
        precisions.append(mt.precision)
        num_labels.append(mt.num_labels)
        num_preds.append(mt.num_preds)
    assert recalls == [
        1 / 1,
        0,
        0,
        0,
        1 / 1,
        0,
        0,
        1 / 1,
        1 / 1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert precisions == [
        1 / 2,
        0,
        0,
        0 / 2,
        1 / 2,
        0,
        0,
        1 / 1,
        1 / 3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert num_labels == [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert num_preds == [2, 0, 0, 2, 2, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
