from collections import Counter, defaultdict
from typing import TypeAlias, cast

Num: TypeAlias = int | float | bool
DEFAULT_KEY = [
    "optim-optimizer-Adadelta",
    "optim-optimizer-Adagrad",
    "optim-optimizer-Adam",
    "optim-optimizer-AdamW",
    "optim-optimizer-SparseAdam",
    "optim-optimizer-Adamax",
    "optim-optimizer-ASGD",
    "optim-optimizer-LBFGS",
    "optim-optimizer-NAdam",
    "optim-optimizer-RAdam",
    "optim-optimizer-RMSprop",
    "optim-optimizer-Rprop",
    "optim-optimizer-SGD",
    "optim-optimizer-MomentumSGD",
    "optim-optimizer-momentum",
    "optim-learningrate",
    "optim-weightdecay",
    "optim-lrscheduler-LambdaLR",
    "optim-lrscheduler-MultiplicativeLR",
    "optim-lrscheduler-StepLR",
    "optim-lrscheduler-MultiStepLR",
    "optim-lrscheduler-ConstantLR",
    "optim-lrscheduler-LinearLR",
    "optim-lrscheduler-ExponentialLR",
    "optim-lrscheduler-PolynomialLR",
    "optim-lrscheduler-CosineAnnealingLR",
    "optim-lrscheduler-ChainedScheduler",
    "optim-lrscheduler-SequentialLR",
    "optim-lrscheduler-ReduceLROnPlateau",
    "optim-lrscheduler-CyclicLR",
    "optim-lrscheduler-OneCycleLR",
    "optim-lrscheduler-CosineAnnealingWarmRestarts",
    "optim-earlystopping",
    "batchsize",
    "iterations",
    "epochs",
    "FPS",
    "runtime-train",
    "runtime-inference",
    "resource-train-gpu-V100",
    "resource-train-gpu-T4",
    "resource-train-gpu-P100",
    "resource-train-gpu-A100",
    "resource-train-gpu-M40",
    "resource-train-tpu-v3",
    "resource-train-tpu-v4",
    "resource-train-gpu-GTX1660Ti",
    "resource-train-gpu-RTX2070",
    "resource-train-gpu-RTX3080",
    "resource-train-gpu-num",
    "resource-train-gpu-memory",
    "resource-inference-gpu-V100",
    "resource-inference-gpu-T4",
    "resource-inference-gpu-P100",
    "resource-inference-gpu-A100",
    "resource-inference-gpu-M40",
    "resource-inference-tpu-v3",
    "resource-inference-tpu-v4",
    "resource-inference-gpu-GTX1660Ti",
    "resource-inference-gpu-RTX2070",
    "resource-inference-gpu-RTX3080",
    "resource-inference-gpu-num",
    "resource-inference-gpu-memory",
]


def clean_values(
    dct: dict[str, Num],
    key_lst: list[str] = DEFAULT_KEY,
) -> dict[str, Num]:
    """
    Cleans a dictionary by removing any keys with None or False values and any keys that are not included in key_lst.

    Parameters
    ----------
    dct : dict[str, Num]
        A dictionaly to be cleaned.
    key_lst : list[str], optional
        A list of keys. The default is DEFAULT_KEY.

    Raises
    ------
    ValueError
        If a value in the input dictionary is not a bool, int, or float.

    Returns
    -------
    dict[str, Num]
        A cleaned dictionary containing only bool, int, or float values.

    """
    assert isinstance(dct, dict), "'dct' must be a dict"
    ret_dct: dict[str, Num] = {}

    for k in key_lst:
        if k not in dct:
            continue
        v = dct[k]
        if v is None or v is False:
            continue
        if isinstance(v, int | float | bool):
            ret_dct[k] = v
        else:
            raise ValueError("Value must be a bool, int, or float")

    return ret_dct


def categorize_dict_keys(
    results: list[dict[str, Num]] | dict[str, Num],
    label_category: dict[str, list[str]],
    vote_option: str | dict = "union",
    keys: list[str] | None = None,
) -> dict[str, set[str | Num]]:
    """
    Categorizes and aggregates dictionaries based on specified label categories and voting options.

    Parameters
    ----------
    results : list[dict[str, Num]] | dict[str, Num]
        A dictionary or a list of dictionaries containing key-value pairs to be categorized.
    label_category : dict[str, list[str]]
        A dictionary mapping category names to lists of keys that belong to each category.
    vote_option : str | dict, optional
        The voting method to use for aggregation. Supported options are "union" and "majority_vote".
        When a str given, it's applied to all categories. When a dict is given, vote options are applied to the corresponding categories.
        The default is "union".
    keys : list[str] | None, optional
        A list of keys. The default is None.

    Raises
    ------
    ValueError
        If the provided `vote_option` is not based on supported voting methods or if 'vote_option' doesn't include all possible keys.

    Returns
    -------
    aggregated_result : dict[str, set[str | Num]]
        A dictionary where each key corresponds to a category name, and the corresponding value is a set containing
        the aggregated values from the keys belonging to that category.

    """
    merged_result: defaultdict = defaultdict(list)
    results_lst: list[dict[str, Num]] = (
        [results] if isinstance(results, dict) else results
    )
    for result in results_lst:
        if keys is None:
            cleaned_result = clean_values(result)
        else:
            cleaned_result = clean_values(result, keys)

        for category, subs in label_category.items():
            is_single_category = True if len(subs) == 1 else False
            for sub in subs:
                if sub not in cleaned_result:
                    continue
                assert isinstance(cleaned_result[sub], int | float | bool)
                if is_single_category:
                    merged_result[category].append(cleaned_result[sub])
                else:
                    merged_result[category].append(sub)

    if isinstance(vote_option, str):
        vote_option = {k: vote_option for k in merged_result.keys()}

    aggregated_result: dict[str, set[str | Num]] = dict()
    for c, v in merged_result.items():
        if c not in vote_option:
            raise ValueError(
                f"Invalid value: 'vote_option'\n'{c}' doesn't exist in 'vote_option' as a key."
            )
        match vote_option[c]:
            case "union":
                aggregated_result[c] = set(v)
            case "majority_vote":
                cnt = Counter(v)
                max_cnt = max(cnt.values())
                majorities = [k for k, v in cnt.items() if v == max_cnt]
                aggregated_result[c] = set(majorities)
            case _:
                raise ValueError(
                    f"'vote_option' must be based on supported voting methods, got '{vote_option[c]}.'"
                )

    return aggregated_result


def uncategorize_dict_keys(
    categorized_dct: dict[str, set[str | Num]],
    label_category: dict[str, list[str]],
) -> dict[str, bool | list[int | float]]:
    """
    Uncategorizes dictionary keys besed on specified label categories.

    Parameters
    ----------
    categorized_dct : dict[str, set[str | Num]]
        A dictionary containing categorized data where keys are category names and values are sets
        containing elements of either string or numeric types.
    label_category : dict[str, list[str]]
       A dictionary mapping category names to lists of subcategory names. This helps determine whether
       a category is a single subcategory or has multiple subcategories.

    Returns
    -------
    uncategorized_dct : dict[str, bool | list[int | float]]
        A dictionary with keys representing either categories or subcategories.

    """
    assert isinstance(categorized_dct, dict)
    for i in categorized_dct.values():
        assert isinstance(i, set)
        for j in i:
            assert isinstance(j, str | Num)

    uncategorized_dct: dict[str, bool | list[int | float]] = dict()
    for category, subs in label_category.items():
        is_single_category = True if len(subs) == 1 else False
        if category not in categorized_dct or not categorized_dct[category]:
            continue
        if is_single_category:
            value = cast(set[Num], categorized_dct[category])
            if len(value) == 1 and list(value)[0] is True:
                uncategorized_dct[category] = True
            else:
                uncategorized_dct[category] = list(value)
        else:
            for sub in subs:
                if sub in categorized_dct[category]:
                    uncategorized_dct[sub] = True

    return uncategorized_dct
