import json
import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import MetricGroup
from gptsenpy.utils import categorize_labels

with open("tests/data/label_category.json", "r") as f:
    LABEL_CATEGORY = json.load(f)


def calc_f1(recall, precision):
    if recall + precision:
        return 2 * (recall * precision) / (recall + precision)
    else:
        return 0


def test_MetricGroup_0():
    data_path = "tests/data/annotation_format.json"
    data = categorize_labels(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)
    mt()

    expected_recall_precision_dct = {i: None for i in LABEL_CATEGORY}
    expected_num_dct = {i: 0 for i in LABEL_CATEGORY}
    expected_export = {
        "recall": 0,
        "precision": 0,
        "f1": 0,
        "num_labels": 0,
        "num_preds": 0,
    }

    assert mt.recall == mt.precision == mt.f1 == 0
    assert mt.num_labels == mt.num_preds == 0
    assert mt.recall_dct == mt.precision_dct == expected_recall_precision_dct
    assert mt.num_labels_dct == mt.num_preds_dct == expected_num_dct
    assert mt.export_metrics() == expected_export


def test_MetricGroup_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_labels(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)
    mt()

    expected_recall_precision_dct = {
        "optim-optimizer": 1,
        "optim-optimizer-momentum": 1,
        "optim-learningrate": 1,
        "optim-weightdecay": 1,
        "optim-lrscheduler": 1,
        "batchsize": 1,
        "epochs": 1,
        "resource-train-gpu": 1,
        "resource-inference-gpu": 1,
    }
    for i in LABEL_CATEGORY:
        if i not in expected_recall_precision_dct:
            expected_recall_precision_dct[i] = None
    expected_num_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
    expected_export = {
        "recall": 9 / 9,
        "precision": 9 / 9,
        "f1": calc_f1(9 / 9, 9 / 9),
        "num_labels": 9,
        "num_preds": 9,
    }

    assert mt.recall == mt.precision == 9 / 9
    assert mt.f1 == calc_f1(9 / 9, 9 / 9)
    assert mt.num_labels == mt.num_preds == 9
    assert mt.recall_dct == mt.precision_dct == expected_recall_precision_dct
    assert mt.num_labels_dct == mt.num_preds_dct == expected_num_dct
    assert mt.export_metrics() == expected_export


def test_MetricGroup_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_labels(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, {}, LABEL_CATEGORY)
    mt()

    expected_recall_dct = {
        "optim-optimizer": 0,
        "optim-optimizer-momentum": 0,
        "optim-learningrate": 0,
        "optim-weightdecay": 0,
        "optim-lrscheduler": 0,
        "batchsize": 0,
        "epochs": 0,
        "resource-train-gpu": 0,
        "resource-inference-gpu": 0,
    }
    for i in LABEL_CATEGORY:
        if i not in expected_recall_dct:
            expected_recall_dct[i] = None
    expected_precision_dct = {i: None for i in LABEL_CATEGORY}
    expected_num_labels_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
    expected_num_preds_dct = {i: 0 for i in LABEL_CATEGORY}
    expected_export = {
        "recall": 0,
        "precision": 0,
        "f1": 0,
        "num_labels": 9,
        "num_preds": 0,
    }

    assert mt.recall == mt.precision == mt.f1 == 0
    assert mt.num_labels == 9
    assert mt.num_preds == 0
    assert mt.recall_dct == expected_recall_dct
    assert mt.precision_dct == expected_precision_dct
    assert mt.num_labels_dct == expected_num_labels_dct
    assert mt.num_preds_dct == expected_num_preds_dct
    assert mt.export_metrics() == expected_export


def test_MetricGroup_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_labels(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup({}, data, LABEL_CATEGORY)
    mt()

    expected_recall_dct = {i: None for i in LABEL_CATEGORY}
    expected_precision_dct = {
        "optim-optimizer": 0,
        "optim-optimizer-momentum": 0,
        "optim-learningrate": 0,
        "optim-weightdecay": 0,
        "optim-lrscheduler": 0,
        "batchsize": 0,
        "epochs": 0,
        "resource-train-gpu": 0,
        "resource-inference-gpu": 0,
    }
    for i in LABEL_CATEGORY:
        if i not in expected_precision_dct:
            expected_precision_dct[i] = None
    expected_num_labels_dct = {i: 0 for i in LABEL_CATEGORY}
    expected_num_preds_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
    expected_export = {
        "recall": 0,
        "precision": 0,
        "f1": 0,
        "num_labels": 0,
        "num_preds": 9,
    }

    assert mt.recall == mt.precision == mt.f1 == 0
    assert mt.num_labels == 0
    assert mt.num_preds == 9
    assert mt.recall_dct == expected_recall_dct
    assert mt.precision_dct == expected_precision_dct
    assert mt.num_labels_dct == expected_num_labels_dct
    assert mt.num_preds_dct == expected_num_preds_dct
    assert mt.export_metrics() == expected_export


def test_MetricGroup_4():
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

    mt = MetricGroup(gts, preds, LABEL_CATEGORY)
    mt()

    expected_recall_dct = {
        "optim-optimizer": 1 / 1,
        "optim-lrscheduler": 1 / 1,
        "optim-earlystopping": 0 / 1,
        "batchsize": 0 / 1,
        "iterations": 1 / 1,
        "epochs": 1 / 1,
    }
    expected_precision_dct = {
        "optim-optimizer": 1 / 2,
        "optim-weightdecay": 0 / 2,
        "optim-lrscheduler": 1 / 2,
        "iterations": 1 / 1,
        "epochs": 1 / 3,
    }
    for i in LABEL_CATEGORY:
        if i not in expected_recall_dct:
            expected_recall_dct[i] = None
        if i not in expected_precision_dct:
            expected_precision_dct[i] = None
    expected_num_labels_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    }
    expected_num_preds_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [2, 0, 0, 2, 2, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    }
    expected_export = {
        "recall": 4 / 6,
        "precision": (1 / 2 + 1 / 2 + 1 / 1 + 1 / 3) / 5,
        "f1": calc_f1(4 / 6, (1 / 2 + 1 / 2 + 1 / 1 + 1 / 3) / 5),
        "num_labels": 6,
        "num_preds": 10,
    }

    assert mt.recall == 4 / 6
    assert mt.precision == (1 / 2 + 1 / 2 + 1 / 1 + 1 / 3) / 5
    assert mt.f1 == calc_f1(4 / 6, (1 / 2 + 1 / 2 + 1 / 1 + 1 / 3) / 5)
    assert mt.num_labels == 6
    assert mt.num_preds == 10
    assert mt.recall_dct == expected_recall_dct
    assert mt.precision_dct == expected_precision_dct
    assert mt.num_labels_dct == expected_num_labels_dct
    assert mt.num_preds_dct == expected_num_preds_dct
    assert mt.export_metrics() == expected_export


def test_MetricGroup_5():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_labels(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, [])
    mt()

    expected_export = {
        "recall": 0,
        "precision": 0,
        "f1": 0,
        "num_labels": 0,
        "num_preds": 0,
    }

    assert mt.recall == mt.precision == mt.f1 == 0
    assert mt.num_labels == mt.num_preds == 0
    assert (
        mt.recall_dct == mt.precision_dct == mt.num_labels_dct == mt.num_preds_dct == {}
    )
    assert mt.export_metrics() == expected_export
