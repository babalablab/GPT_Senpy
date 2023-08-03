import json
import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import MetricGroup
from gptsenpy.utils import categorize_dict_keys


with open("tests/data/label_category.json", "r") as f:
    LABEL_CATEGORY = json.load(f)


def calc_f1(recall, precision):
    if recall + precision:
        return 2 * (recall * precision) / (recall + precision)
    else:
        return 0


def test_Metrics_0():
    data_path = "tests/data/annotation_format.json"
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)

    expected_metrics = {i: 0 for i in LABEL_CATEGORY}
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
        mt.recall_dct
        == mt.precision_dct
        == mt.f1_dct
        == mt.num_labels_dct
        == mt.num_preds_dct
        == expected_metrics
    )
    assert mt.export_metrics() == expected_export


def test_Metrics_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)

    expected_recall_precision = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
    expected_f1_dct = {
        i: calc_f1(j, j)
        for i, j in zip(LABEL_CATEGORY, expected_recall_precision.values())
    }
    expected_num = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
    expected_export = {
        "recall": 9 / 18,
        "precision": 9 / 18,
        "f1": sum(expected_f1_dct.values()) / len(expected_f1_dct.values()),
        "num_labels": 9,
        "num_preds": 9,
    }

    assert mt.recall == mt.precision == 9 / 18
    assert mt.f1 == sum(expected_f1_dct.values()) / len(expected_f1_dct.values())
    assert mt.num_labels == mt.num_preds == 9
    assert mt.recall_dct == mt.precision_dct == expected_recall_precision
    assert mt.f1_dct == expected_f1_dct
    assert mt.num_labels_dct == mt.num_preds_dct == expected_num
    assert mt.export_metrics() == expected_export


def test_Metrics_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, {}, LABEL_CATEGORY)

    expected_metrics = {i: 0 for i in LABEL_CATEGORY}
    expected_num_labels_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY, [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
        )
    }
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
    assert (
        mt.recall_dct
        == mt.precision_dct
        == mt.f1_dct
        == mt.num_preds_dct
        == expected_metrics
    )
    assert mt.num_labels_dct == expected_num_labels_dct
    assert mt.export_metrics() == expected_export


def test_Metrics_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup({}, data, LABEL_CATEGORY)

    expected_metrics = {i: 0 for i in LABEL_CATEGORY}
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
    assert (
        mt.recall_dct
        == mt.precision_dct
        == mt.f1_dct
        == mt.num_labels_dct
        == expected_metrics
    )
    assert mt.num_preds_dct == expected_num_preds_dct
    assert mt.export_metrics() == expected_export


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

    mt = MetricGroup(gts, preds, LABEL_CATEGORY)

    expected_recall_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY,
            [1 / 1, 0, 0, 0, 1 / 1, 0, 0, 1 / 1, 1 / 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
    }
    expected_precision_dct = {
        i: j
        for i, j in zip(
            LABEL_CATEGORY,
            [1 / 2, 0, 0, 0 / 2, 1 / 2, 0, 0, 1 / 1, 1 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
    }
    expected_f1_dct = {
        i: calc_f1(j, k)
        for i, j, k in zip(
            LABEL_CATEGORY,
            expected_recall_dct.values(),
            expected_precision_dct.values(),
        )
    }
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
        "recall": 4 / 18,
        "precision": (2 + 1 / 3) / 18,
        "f1": sum(expected_f1_dct.values()) / len(expected_f1_dct.values()),
        "num_labels": 6,
        "num_preds": 10,
    }

    assert mt.recall == 4 / 18
    assert mt.precision == (2 + 1 / 3) / 18
    assert mt.f1 == sum(expected_f1_dct.values()) / len(expected_f1_dct.values())
    assert mt.num_labels == 6
    assert mt.num_preds == 10
    assert mt.recall_dct == expected_recall_dct
    assert mt.precision_dct == expected_precision_dct
    assert mt.f1_dct == expected_f1_dct
    assert mt.num_labels_dct == expected_num_labels_dct
    assert mt.num_preds_dct == expected_num_preds_dct
    assert mt.export_metrics() == expected_export


def test_Metrics_5():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, [])

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
        mt.recall_dct
        == mt.precision_dct
        == mt.f1_dct
        == mt.num_labels_dct
        == mt.num_preds_dct
        == {}
    )
    assert mt.export_metrics() == expected_export
