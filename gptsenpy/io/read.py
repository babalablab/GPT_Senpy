import json
import os
import sys
from pathlib import Path
from typing import Dict, Union


def read_json(path: Union[Path, str]) -> Dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_text(path: Union[Path, str]) -> str:
    with open(path, "r") as f:
        data = f.read()
    data = data.rstrip()
    return data
