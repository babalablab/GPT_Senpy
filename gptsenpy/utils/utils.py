from typing import Any, Optional, TypeAlias

Num: TypeAlias = int | float


def clean_values(
    values: dict[str, bool | list[Num] | set[Num]]
) -> dict[str, bool | set[Num]]:
    """
    Cleans a dictionary of values by removing any keys with None or False values, and converting
    any lists or floats to sets. The resulting dictionary will only contain keys with boolean or
    set values.

    Args:
        values (dict[str, bool | list | set]): A dictionary of values to be cleaned.

    Returns:
        dict[str, bool | set[Num]]: A cleaned dictionary containing only boolean or set values.

    Raises:
        ValueError: If a value in the input dictionary is not a bool, list, int, or float.

    Examples:
        >>> clean_values({'a': True, 'b': False, 'c': [1, 2, 3], 'd': 4.5, 'e': None})
        {'a': True, 'c': {1, 2, 3}, 'd': {4.5}}

    """
    assert isinstance(values, dict), "Values must be a dict"
    ret_dict: dict[str, bool | set[Num]] = {}

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


def get_denominator(values: dict[str, bool | set[Num]]) -> int:
    """
    Calculates the denominator for a fraction based on the given dictionary of values.

    Args:
        values (dict[str, bool | set[Num]]): A dictionary of values where the keys are strings and the values are either boolean or sets of numbers.

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


def concat_json_result(results: list[dict[str, Any]]) -> dict[str, set[Num] | bool]:
    merged_dict = {}
    for result in results:
        cleaned_result = clean_values(result)

        for k, v in cleaned_result.items():
            if k not in merged_dict:
                merged_dict[k] = v
            elif isinstance(v, bool):
                merged_dict[k] = merged_dict[k] or v
            elif isinstance(v, set):
                merged_dict[k] = merged_dict[k] | v  # type: ignore
            else:
                raise TypeError("Unexpected type in result")
        merged_dict = clean_values(merged_dict)  # type: ignore
    return merged_dict
