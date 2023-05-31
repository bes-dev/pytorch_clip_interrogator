"""
Copyright 2023 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from typing import List, Optional, Dict
import torch


def to_list(var):
    if not isinstance(var, list):
        var = [var]
    return var


def to_device(
        var: Dict[str, torch.Tensor],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """ Move all tensors from dict to specific device/dtype.

    Args:
        var (Dict[str, torch.Tensor]): dictionary of torch.Tensor.
        device (str): target device (default: "cpu").
        dtype (Optional[torch.dtype]): target data type (default: None)
    Returns:
        dict of tensors.
    """
    for v, k in var.items():
        var[v] = k.to(device=device, dtype=dtype)
    return var


def get_module_path():
    """ Get module path
    Returns:
        path (str): path to current module.
    """
    file_path = os.path.abspath(__file__)
    module_path = os.path.dirname(file_path)
    return module_path


def res_path(path):
    """ Resource path
    Arguments:
        path (str): related path from module dir to some resources.
    Returns:
        path (str): absolute path to module dir.
    """
    return os.path.join(get_module_path(), path)


def improve_prompts(
        prompts: List[str],
        pattern: str = "<p>"
) -> List[str]:
    """ Inprove prompts by pattern.

    Args:
        prompts (List[str]): list of prompts to improve.
        pattern (str): pattern.

    Returns:
        Improved prompts (List[str])

    Examples:
        improve(["Van Gogh", "Claude Monet"], pattern="inspired by <p>")
    """
    out = []
    for prompt in prompts:
        out.append(pattern.replace("<p>", prompt))
    return out


def load_prompts(path: str, patterns: Optional[List[str]] = None) -> List[str]:
    """ Load prompts from file.

    Args:
        path (str): path to prompts file.
        patterns (Optional[List[str]]): list of patterns to improve prompts.

    Returns:
        List of prompts.

    Examples:
        load_prompts("data/artists.txt", patterns=["inspired by <p>", "by <p>"])
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    prompts = items.copy()
    if patterns is not None:
        for pattern in patterns:
            prompts.extend(improve_prompts(items, pattern))
    return prompts
