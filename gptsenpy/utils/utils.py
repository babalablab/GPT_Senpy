from typing import Any, TypeAlias


Num: TypeAlias = int | float
DEFAULT_KEY = ['optim-optimizer-Adadelta',
               'optim-optimizer-Adagrad',
               'optim-optimizer-Adam',
               'optim-optimizer-AdamW',
               'optim-optimizer-SparseAdam',
               'optim-optimizer-Adamax',
               'optim-optimizer-ASGD',
               'optim-optimizer-LBFGS',
               'optim-optimizer-NAdam',
               'optim-optimizer-RAdam',
               'optim-optimizer-RMSprop',
               'optim-optimizer-Rprop',
               'optim-optimizer-SGD',
               'optim-optimizer-MomentumSGD',
               'optim-optimizer-MomentumSGD-momentum',
               'optim-learningrate',
               'optim-weightdecay',
               'optim-lrscheduler-LambdaLR',
               'optim-lrscheduler-MultiplicativeLR',
               'optim-lrscheduler-StepLR',
               'optim-lrscheduler-MultiStepLR',
               'optim-lrscheduler-ConstantLR',
               'optim-lrscheduler-LinearLR',
               'optim-lrscheduler-ExponentialLR',
               'optim-lrscheduler-PolynomialLR',
               'optim-lrscheduler-CosineAnnealingLR',
               'optim-lrscheduler-ChainedScheduler',
               'optim-lrscheduler-SequentialLR',
               'optim-lrscheduler-ReduceLROnPlateau',
               'optim-lrscheduler-CyclicLR',
               'optim-lrscheduler-OneCycleLR',
               'optim-lrscheduler-CosineAnnealingWarmRestarts',
               'optim-earlystopping',
               'batchsize',
               'iterations',
               'epochs',
               'FPS',
               'runtime-train',
               'runtime-inference',
               'resource-train-gpu-V100',
               'resource-train-gpu-T4',
               'resource-train-gpu-P100',
               'resource-train-gpu-A100',
               'resource-train-gpu-M40',
               'resource-train-tpu-v3',
               'resource-train-tpu-v4',
               'resource-train-gpu-GTX1660Ti',
               'resource-train-gpu-RTX2070',
               'resource-train-gpu-RTX3080',
               'resource-train-gpu-num']


def clean_values(
    values: dict[str, bool | list[Num] | set[Num]],
    key_lst: list[str] = None,
) -> dict[str, bool | set[Num]]:
    """
    Cleans a dictionary of values by removing any keys with None or False values, and converting
    any lists or floats to sets. The resulting dictionary will only contain keys with boolean or
    set values.

    Args:
        values (dict[str, bool | list | set]): A dictionary of values to be cleaned.

        key_lst (list[str]): A list of keys. 'DEFAULT_KEY' is used by default.

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

    keys = DEFAULT_KEY if key_lst is None else key_lst
    for k in keys:
        if k not in values:
            continue
        v = values[k]
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
    """
    Merges a list of dictionaries into a single dictionary.

    The function iterates over each dictionary in the input list. It cleans the values of each
    dictionary using the `clean_values` function. Then, for each key-value pair in the cleaned
    dictionary, if the key is not already in the merged dictionary, it adds the key-value pair.
    If the key is already present, it performs a logical OR operation for boolean values, or a set
    union operation for set values.

    If a value is not a boolean or a set, it raises a TypeError.

    The merged dictionary is cleaned with `clean_values` function at the end of each iteration.

    Args:
        results: A list of dictionaries. The dictionaries should contain keys of type str and
                 values of type set of numbers (Num) or bool.

    Returns:
        A merged dictionary with keys of type str and values of type set of numbers (Num) or bool.

    Raises:
        TypeError: If a value in the dictionary is not a boolean or a set.
    """
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
