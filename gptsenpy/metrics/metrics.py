from typing import Optional


def clean_values(
    values: dict[str, bool | list[int | float] | set[int | float]]
) -> dict[str, bool | set[int | float]]:
    """
    Cleans a dictionary of values by removing any keys with None or False values, and converting
    any lists or floats to sets. The resulting dictionary will only contain keys with boolean or
    set values.

    Args:
        values (dict[str, bool | list | set]): A dictionary of values to be cleaned.

    Returns:
        dict[str, bool | set[int | float]]: A cleaned dictionary containing only boolean or set values.

    Raises:
        ValueError: If a value in the input dictionary is not a bool, list, int, or float.

    Examples:
        >>> clean_values({'a': True, 'b': False, 'c': [1, 2, 3], 'd': 4.5, 'e': None})
        {'a': True, 'c': {1, 2, 3}, 'd': {4.5}}

    """
    assert isinstance(values, dict), "Values must be a dict"
    ret_dict: dict[str, bool | set[int | float]] = {}

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
            raise ValueError("Value must be a bool, set, list, int, or float")

    return ret_dict


def get_denominator(values: dict[str, bool | set[int | float]]) -> int:
    """
    Calculates the denominator for a fraction based on the given dictionary of values.

    Args:
        values (dict[str, bool | set]): A dictionary of values where the keys are strings and the values are either boolean or sets of strings.

    Returns:
        int: The denominator for the fraction, which is the sum of the number of values in each set and the number of boolean values.

    Raises:
        TypeError: If the input values are not in the expected format.

    Example:
        >>> get_denominator({'a': True, 'b': False, 'c': {'d', 'e', 'f'}})
        4
    """
    ret = 0
    for v in values.values():
        if isinstance(v, set):
            ret += len(v)
        else:
            ret += 1
    return ret


class Metrics:
    def __init__(
        self,
        labels: dict[str, bool | set[int | float] | list[int | float]],
        preds: dict[str, bool | set[int | float] | list[int | float]],
    ):
        self.labels = clean_values(labels)
        self.preds = clean_values(preds)

        self.precision: float = self.get_precision()
        self.recall: float = self.get_recall()
        self.f1: float = self.get_f1()

    def get_recall(
        self,
        labels: Optional[dict[str, bool | set[int | float]]] = None,
        preds: Optional[dict[str, bool | set[int | float]]] = None,
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
        labels: Optional[dict[str, bool | set[int | float]]] = None,
        preds: Optional[dict[str, bool | set[int | float]]] = None,
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
        labels: Optional[dict[str, bool | set[int | float]]] = None,
        preds: Optional[dict[str, bool | set[int | float]]] = None,
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
    ) -> dict[str, bool | set[int | float]]:
        if values is None and name is not None:
            return getattr(self, name)
        elif values is not None:
            return clean_values(values)
        else:
            raise ValueError("Values and name cannot be None")
