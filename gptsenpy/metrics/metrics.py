from typing import Any

from ipdb import set_trace as ist


def clean_values(values: dict[str, Any]) -> dict[str, Any]:
    """
    remove key from dict if value is null
    Args:
        values: annotation infomations

    Returns:
        anotaion information without null values

    """
    ret_dict = {}
    for k, v in values.items():
        if v is not None:
            ret_dict[k] = v
    return ret_dict


class Metrics:
    def __init__(self, labels: dict[str, Any], preds: dict[str, Any]):
        self.labels = clean_values(labels)
        self.preds = clean_values(preds)

        self.precision: float = self.get_precision()
        self.recall: float = self.get_recall()
        self.f1: float = self.get_f1()

    def get_precision(self) -> float:
        preds = self.preds
        labels = self.labels
        if len(labels) == 0:
            return 0.0
        else:
            numerator = sum(
                [
                    1 if k in preds.keys() and preds[k] == v else 0
                    for k, v in labels.items()
                ]
            )
        return numerator / len(labels)

    def get_recall(self) -> float:
        preds = self.preds
        labels = self.labels
        if len(preds) == 0:
            return 0.0
        else:
            numerator = sum(
                [
                    1 if k in labels.keys() and labels[k] == v else 0
                    for k, v in preds.items()
                ]
            )
        return numerator / len(preds)

    def get_f1(self) -> float:
        recall, precision = self.recall, self.precision
        if recall + precision == 0:
            return 0
        return 2 * recall * precision / (recall + precision)
