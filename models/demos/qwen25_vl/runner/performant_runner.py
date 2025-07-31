# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.qwen25_vl.runner.performant_runner_infra import Qwen25VLPerformanceRunnerInfra


class Qwen25VLPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=1,
        seq_len=14308,
        image_grid_thw=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
    ):
        self.device = device
        self.device_batch_size = device_batch_size
        self.seq_len = seq_len
        self.image_grid_thw = image_grid_thw
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype

        self.runner_infra = Qwen25VLPerformanceRunnerInfra(
            device=device,
            batch_size=device_batch_size,
            seq_len=seq_len,
            image_grid_thw=image_grid_thw,
            act_dtype=act_dtype,
            weight_dtype=weight_dtype,
        )

        (
            self.pt_patch_input,
            self.tt_inputs,
            self.input_mem_config,
            self.window_index,
            self.cu_seqlens,
            self.cu_window_seqlens,
            self.rot_mats,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        # self.tt_inputs_host_ttnn = ttnn.from_torch(self.tt_inputs_host, device=self.device)
        # self.tt_inputs_host_ttnn = ttnn.from_torch(self.tt_inputs_host)
        # self.tt_inputs = ttnn.from_torch(self.tt_inputs_host, device=device)
        self.tt_inputs_host = self.runner_infra.get_host_tensor()

    def _capture_qwen25vl_trace_2cqs(self):
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(1, self.op_event)
        # Create fresh device tensor for first copy
        self.tt_inputs = ttnn.allocate_tensor_on_device(self.tt_inputs_host.spec, self.device)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        spec_input = self.runner_infra.ttnn_input.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()
        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        # Create fresh device tensor for second copy
        self.tt_inputs = ttnn.allocate_tensor_on_device(self.tt_inputs_host.spec, self.device)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        # Capture
        ttnn.wait_for_event(1, self.op_event)
        # Create fresh device tensor for third copy
        self.tt_inputs = ttnn.allocate_tensor_on_device(self.tt_inputs_host.spec, self.device)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.ttnn_input.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()  # Do NOT call validate during trace capture
        self.ttnn_input = ttnn.allocate_tensor_on_device(spec_input, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        assert trace_input_addr == self.ttnn_input.buffer_address()

    def _execute_qwen25vl_trace_2cqs_inference(self, tt_inputs_host=None):
        if tt_inputs_host is None:
            tt_inputs_host = self.tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.ttnn_input = ttnn.reshard(self.tt_inputs, self.input_mem_config, self.ttnn_input)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        return self.runner_infra.ttnn_output_tensor[0]

    def _validate(self, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output
        from tests.ttnn.utils_for_testing import assert_with_pcc

        assert_with_pcc(torch_output_tensor, result_output_tensor, self.runner_infra.valid_pcc)

    def release(self):
        ttnn.release_trace(self.device, self.tid)

    def run(self, input_tensor=None):
        tt_inputs_host = self.runner_infra.setup_l1_sharded_input(input_tensor)
        output = self._execute_qwen25vl_trace_2cqs_inference(tt_inputs_host)
        return output
