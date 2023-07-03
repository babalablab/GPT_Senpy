import sys
from typing import Optional, TypeAlias

from ..utils import clean_values, get_denominator

Num: TypeAlias = int | float


class Metrics:
    def __init__(
        self,
        labels: dict[str, bool | set[Num] | list[Num]],
        preds: dict[str, bool | set[Num] | list[Num]],
    ):
        self.labels = clean_values(labels)
        self.preds = clean_values(preds)

        self.precision: float = self.get_precision()
        self.recall: float = self.get_recall()
        self.f1: float = self.get_f1()

    def export_metrics(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def get_recall(
        self,
        labels: Optional[dict[str, bool | set[Num]]] = None,
        preds: Optional[dict[str, bool | set[Num]]] = None,
    ) -> float:
        labels = self.get_value_if_none(labels, "labels")
        preds = self.get_value_if_none(preds, "preds")
        if len(labels) == 0:
            return 0.0

        denominator = get_denominator(labels)
        numerator = 0
        for k, v in labels.items():
            if k not in preds.keys():
                continue
            if isinstance(v, set):
                for vv in v:
                    preds_k = preds[k]
                    if isinstance(preds_k, set) and vv in preds_k:
                        numerator += 1
                        break
            else:
                if preds[k] == v:
                    numerator += 1

        return numerator / denominator

    def get_precision(
        self,
        labels: Optional[dict[str, bool | set[Num]]] = None,
        preds: Optional[dict[str, bool | set[Num]]] = None,
    ) -> float:
        labels = self.get_value_if_none(labels, "labels")
        preds = self.get_value_if_none(preds, "preds")

        if len(preds) == 0:
            return 0.0
        numerator = 0
        denominator = get_denominator(preds)
        for k, v in preds.items():
            if k not in labels.keys():
                continue

            if isinstance(v, set):
                for vv in v:
                    labels_k = labels[k]
                    if isinstance(labels_k, set) and vv in labels_k:
                        numerator += 1
                        break
            else:
                if labels[k] == v:
                    numerator += 1

        return numerator / denominator

    def get_f1(
        self,
        labels: Optional[dict[str, bool | set[Num]]] = None,
        preds: Optional[dict[str, bool | set[Num]]] = None,
    ) -> float:
        labels = self.get_value_if_none(labels, "labels")
        preds = self.get_value_if_none(preds, "preds")
        recall, precision = self.get_recall(labels, preds), self.get_precision(
            labels, preds
        )

        if recall + precision == 0:
            return 0
        return 2 * recall * precision / (recall + precision)

    def get_value_if_none(
        self, values: Optional[dict] = None, name: Optional[str] = None
    ) -> dict[str, bool | set[Num]]:
        if values is None and name is not None:
            return getattr(self, name)
        elif values is not None:
            return clean_values(values)
        else:
            raise ValueError("Values and name cannot be None")
