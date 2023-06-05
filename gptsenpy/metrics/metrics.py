from typing import Any, Optional


def clean_values(values: dict[str, Any]) -> dict[str, Any]:
    """
    remove key from dict if value is null
    Args:
        values: annotation infomations

    Returns:
        anotaion information without null values

    """
    assert isinstance(values, dict), "Values must be a dict"
    return {k: v for k, v in values.items() if v is not None}


class Metrics:
    def __init__(self, labels: dict[str, Any], preds: dict[str, Any]):
        self.labels = clean_values(labels)
        self.preds = clean_values(preds)

        self.precision: float = self.get_precision()
        self.recall: float = self.get_recall()
        self.f1: float = self.get_f1()

    def get_precision(
        self, labels: Optional[dict] = None, preds: Optional[dict] = None
    ) -> float:
        labels = self.get_value_if_none(labels, "labels")
        preds = self.get_value_if_none(preds, "preds")
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

    def get_recall(
        self, labels: Optional[dict] = None, preds: Optional[dict] = None
    ) -> float:
        labels = self.get_value_if_none(labels, "labels")
        preds = self.get_value_if_none(preds, "preds")

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

    def get_f1(
        self, labels: Optional[dict] = None, preds: Optional[dict] = None
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
    ) -> dict[str, Any]:
        if values is None and name is not None:
            return getattr(self, name)
        elif values is not None:
            return clean_values(values)
        else:
            raise ValueError("Values and name cannot be None")
