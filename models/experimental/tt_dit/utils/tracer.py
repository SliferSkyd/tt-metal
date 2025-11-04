# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Hashable
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def tree_map(f: Callable, x: Any, *xs: Any) -> Any:
    """Apply a function to leaves of nested data structures.

    Recursively traverses nested structures (tuples, lists, dicts) and applies
    the given function to corresponding leaf elements across all input structures.

    Args:
        f: A callable that takes N arguments, where N is the number of input
            structures (1 + len(xs)). Applied to leaf elements.
        x: The first nested data structure to traverse.
        *xs: Additional nested data structures with the same shape as x.

    Returns:
        A new nested structure with the same shape as the inputs, where each
        leaf has been transformed by applying f to the corresponding leaves
        from all input structures.

    Raises:
        ValueError: If the input structures don't have matching types at
            corresponding positions, or if dicts have different keys.

    Examples:
        >>> tree_map(lambda a: a * 2, [1, 2, 3])
        [2, 4, 6]

        >>> tree_map(lambda a, b: a + b, [1, 2], [10, 20])
        [11, 22]

        >>> tree_map(lambda a, b: a + b,
        ...          {"a": 1, "b": [2, 3]},
        ...          {"a": 10, "b": [20, 30]})
        {'a': 11, 'b': [22, 33]}
    """
    tx = type(x)
    for y in xs:
        if type(y) is not tx:
            msg = f"types should be the same: {tx} != {type(y)}"
            raise ValueError(msg)

    if isinstance(x, tuple):
        return tuple(tree_map(f, *elts) for elts in zip(x, *xs, strict=True))

    if isinstance(x, list):
        return [tree_map(f, *elts) for elts in zip(x, *xs, strict=True)]

    if isinstance(x, dict):
        for y in xs:
            if x.keys() != y.keys():
                msg = "dict keys should be the same"
                raise ValueError(msg)
        return {key: tree_map(f, *(d[key] for d in (x, *xs))) for key in x}

    return f(x, *xs)


def tree_to_nested_tuples(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(tree_to_nested_tuples(x) for x in value)
    if isinstance(value, dict):
        return tuple((k, tree_to_nested_tuples(v)) for k, v in sorted(value.items()))
    return value


def check(value: Any) -> Any:
    if isinstance(value, (ttnn.Tensor, Hashable)):
        return value

    msg = f"unsupported type: {type(value)}"
    raise TypeError(msg)


def move_to_device_if_tensor(device: ttnn.MeshDevice, value: Any) -> Any:
    if not isinstance(value, ttnn.Tensor):
        return value

    if value.device() is None:
        return value.to(device)

    if value.device() != device:
        msg = f"tensor is on device {value.device()}, expected {device}"
        raise ValueError(msg)

    return value


def copy_if_tensor(src: Any, dst: Any) -> None:
    if not isinstance(src, ttnn.Tensor):
        return

    if src.device() is None:
        ttnn.copy_host_to_device_tensor(src, dst)
    else:
        ttnn.copy(src, dst)


def tensor_to_properties(value: Any) -> Any:
    if not isinstance(value, ttnn.Tensor):
        return value

    return tuple(value.shape), value.dtype, value.layout


# TODO: the biggest issue with this is that all inputs are copied with every call, even it they are
# unchanged.
def autotrace(f: Callable) -> Callable:
    traces = {}

    @functools.wraps(f)
    def wrapper(
        *args: Any,
        traced: bool = False,
        trace_device: ttnn.MeshDevice = None,
        **kwargs: Any,
    ) -> Any:
        if not traced:
            return f(*args, **kwargs)

        if trace_device is None:
            msg = "trace_device must be specified when traced=True"
            raise ValueError(msg)

        inputs = {"args": args, "kwargs": kwargs, "device": trace_device}
        inputs = tree_map(check, inputs)
        inputs_key = tree_to_nested_tuples(tree_map(tensor_to_properties, inputs))

        if inputs_key in traces:
            trace = traces[inputs_key]

            tree_map(copy_if_tensor, inputs, trace["inputs"])
            ttnn.execute_trace(trace_device, trace["id"], cq_id=0, blocking=False)
            return trace["outputs"]

        inputs = tree_map(lambda value: move_to_device_if_tensor(trace_device, value), inputs)

        # compile
        f(*inputs["args"], **inputs["kwargs"])

        # capture trace
        trace_id = ttnn.begin_trace_capture(trace_device, cq_id=0)
        try:
            outputs = f(*inputs["args"], **inputs["kwargs"])
        finally:
            ttnn.end_trace_capture(trace_device, trace_id, cq_id=0)

        traces[inputs_key] = {
            "id": trace_id,
            "inputs": inputs,
            "outputs": tree_map(check, outputs),
        }

        return outputs

    return wrapper
