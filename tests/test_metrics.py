import json
import sys

sys.path.append("../gptsenpy")

from gptsenpy.io.read import read_json
from gptsenpy.metrics import MetricGroup
from gptsenpy.utils import categorize_dict_keys


with open("tests/data/label_category.json", "r") as f:
    LABEL_CATEGORY = json.load(f)


def test_Metrics_0():
    data_path = "tests/data/annotation_format.json"
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)
    assert mt.recall == mt.precision == 0
    assert mt.num_labels == mt.num_preds == 0
    assert (
        mt.recall_lst
        == mt.precision_lst
        == mt.num_labels_lst
        == mt.num_preds_lst
        == [0] * len(LABEL_CATEGORY)
    )

    expected = {"recall": 0, "precision": 0, "num_labels": 0, "num_preds": 0}
    assert mt.export_metrics() == expected


def test_Metrics_1():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, data, LABEL_CATEGORY)
    assert mt.recall == mt.precision == 9 / 18
    assert mt.num_labels == mt.num_preds == 9
    assert (
        mt.recall_lst
        == mt.precision_lst
        == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    )
    assert (
        mt.num_labels_lst
        == mt.num_preds_lst
        == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    )

    expected = {"recall": 9 / 18, "precision": 9 / 18, "num_labels": 9, "num_preds": 9}
    assert mt.export_metrics() == expected


def test_Metrics_2():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup(data, {}, LABEL_CATEGORY)
    assert mt.recall == mt.precision == 0
    assert mt.num_labels == 9
    assert mt.num_preds == 0
    assert (
        mt.recall_lst
        == mt.precision_lst
        == mt.num_preds_lst
        == [0] * len(LABEL_CATEGORY)
    )
    assert mt.num_labels_lst == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]

    expected = {"recall": 0, "precision": 0, "num_labels": 9, "num_preds": 0}
    assert mt.export_metrics() == expected


def test_Metrics_3():
    data_path = (
        "tests/data/DAMO-YOLO_A_Report_on_Real-Time_Object_Detection_Design.json"
    )
    data = categorize_dict_keys(read_json(data_path), LABEL_CATEGORY)

    mt = MetricGroup({}, data, LABEL_CATEGORY)
    assert mt.recall == mt.precision == 0
    assert mt.num_labels == 0
    assert mt.num_preds == 9
    assert (
        mt.recall_lst
        == mt.precision_lst
        == mt.num_labels_lst
        == [0] * len(LABEL_CATEGORY)
    )
    assert mt.num_preds_lst == [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]

    expected = {"recall": 0, "precision": 0, "num_labels": 0, "num_preds": 9}
    assert mt.export_metrics() == expected


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
    assert mt.recall == 4 / 18
    assert mt.precision == (2 + 1 / 3) / 18
    assert mt.num_labels == 6
    assert mt.num_preds == 10
    assert mt.recall_lst == [
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
    assert mt.precision_lst == [
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
    assert mt.num_labels_lst == [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert mt.num_preds_lst == [2, 0, 0, 2, 2, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    expected = {
        "recall": 4 / 18,
        "precision": (2 + 1 / 3) / 18,
        "num_labels": 6,
        "num_preds": 10,
    }
    assert mt.export_metrics() == expected
