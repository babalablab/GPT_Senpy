import json
import os
import sys
from pathlib import Path
from typing import Dict, Union


def check_duplicate_keys(pairs):
    """check duplicate keys in json file
    Args:
        pairs dict(str, Any):
    Returns:
        dict
    """
    assert len(pairs) == len(set([k for k, _ in pairs])), "Duplicate keys in JSON file."
    return dict(pairs)


def read_json(path: Union[Path, str]) -> Dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f, object_pairs_hook=check_duplicate_keys)
    return data


def read_text(path: Union[Path, str]) -> str:
    with open(path, "r") as f:
        data = f.read()
    data = data.rstrip()
    return data
