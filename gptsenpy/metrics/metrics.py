from typing import Any, Optional


def clean_values(values: dict[str, Any]) -> dict[str, Any]:
    """
    Remove keys from the dictionary if the corresponding value is null.

    Args:
        values (dict[str, Any]): Annotation information.

    Returns:
        dict[str, Any]: Annotation information without null values.

    Raises:
        ValueError: If the value is not of type bool, list, int, or float.

    """
    assert isinstance(values, dict), "Values must be a dict"
    ret_dict: dict[str, bool | set | int | float] = {}

    for k, v in values.items():
        if v is None or v is False:
            continue
        if isinstance(v, bool):
            ret_dict[k] = v
        elif isinstance(v, (list, set)):
            ret_dict[k] = set(v)
        elif isinstance(v, (int, float)):
            ret_dict[k] = set([v])
        else:
            raise ValueError("Value must be a bool, list, int, or float")

    return ret_dict


def get_denominator(values: dict[str, Any]) -> int:
    """
    Calculate the denominator value based on the given dictionary.

    Args:
        values (dict[str, Any]): Dictionary containing values.

    Returns:
        int: The calculated denominator value.

    """
    ret = 0
    for v in values.values():
        if isinstance(v, (list, set)):
            ret += len(v)
        else:
            ret += 1
    return ret


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

        denominator = get_denominator(labels)
        numerator = 0
        for k, v in labels.items():
            if k not in preds.keys():
                continue
            if isinstance(v, (list, set)):
                for vv in v:
                    if vv in preds[k]:
                        numerator += 1
                        break
            else:
                if preds[k] == v:
                    numerator += 1

        return numerator / denominator

    def get_recall(
        self, labels: Optional[dict] = None, preds: Optional[dict] = None
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

            if isinstance(v, (list, set)):
                for vv in v:
                    if vv in labels[k]:
                        numerator += 1
                        break
            else:
                if labels[k] == v:
                    numerator += 1

        return numerator / denominator

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
