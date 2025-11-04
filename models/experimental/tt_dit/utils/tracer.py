# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class Tracer:
    def __init__(self, *, device: ttnn.MeshDevice, function: Callable) -> None:
        self._trace_device = device
        self._function = function
        self._trace_tensors = {}
        self._trace_id = None

    def __getitem__(self, key: str) -> ttnn.Tensor:
        if key not in self._trace_tensors:
            msg = f"tensor '{key}' is not set"
            raise KeyError(msg)

        return self._trace_tensors[key]

    def __setitem__(self, key: str, value: ttnn.Tensor) -> None:
        if value.device() is not None and value.device() != self._trace_device:
            msg = f"tensor '{key}' must be on device {self._trace_device}, but got {value.device()}"
            raise ValueError(msg)

        if self._trace_id is None:
            if value.device() is None:
                value = value.to(self._trace_device)

            self._trace_tensors[key] = value
            return

        if key not in self._trace_tensors:
            msg = f"tensor '{key}' was not set before tracing"
            raise ValueError(msg)

        if value.device() is None:
            ttnn.copy_host_to_device_tensor(value, self._trace_tensors[key])
        else:
            ttnn.copy(value, self._trace_tensors[key])

    def capture_or_execute_trace(
        self, *, cq_id: int = 0, args: Sequence[Any] = (), kwargs: Mapping[str, Any] | None = None
    ) -> None:
        if kwargs is None:
            kwargs = {}

        if self._trace_id is not None:
            ttnn.execute_trace(self._trace_device, self._trace_id, cq_id=cq_id, blocking=False)
            return

        self._function(*args, **kwargs)

        trace_id = ttnn.begin_trace_capture(self._trace_device, cq_id=cq_id)
        try:
            self._function(*args, **kwargs)
        finally:
            ttnn.end_trace_capture(self._trace_device, trace_id, cq_id=cq_id)

        self._trace_id = trace_id
