# SPDX-FileCopyrightText: © 2025 T        self.valid_pcc = 0.6  # Based on observed PCC valuesnstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.demos.qwen25_vl.tt.model import VisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys,
)
from models.utility_functions import comp_pcc


class Qwen25VLPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        seq_len,
        image_grid_thw=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_grid_thw = image_grid_thw if image_grid_thw is not None else torch.tensor([[1, 98, 146]])
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.valid_pcc = 0.6  # Based on observed PCC values
        self.model_args = VisionModelArgs(device, dummy_weights=True, max_batch_size=batch_size, max_seq_len=seq_len)
        self.reference_model = self.model_args.reference_vision_model(
            depth=self.model_args.hf_config.vision_config.depth
        )
        state_dict = standardize_hf_keys(self.reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, self.model_args.head_dim)
        state_dict_prefix = self.model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}
        self.tt_model = VisionTransformer(
            args=self.model_args,
            state_dict=state_dict,
            weight_cache_path=self.model_args.weight_cache_path(act_dtype),
            dtype=act_dtype,
        )
        self.pt_pixel_values = torch.randn([14308, 1176]) * 0.8320 + 1.2969
        (
            cu_seqlens_torch,
            cu_window_seqlens_torch,
            position_embeddings,
            self.window_index,
        ) = qwen2_5_vision_transformer_preprocess(
            seq_len=14308,
            grid_thw=self.image_grid_thw,
            head_dim=self.model_args.head_dim,
            spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            window_size=self.model_args.hf_config.vision_config.window_size,
            patch_size=self.model_args.hf_config.vision_config.patch_size,
        )
        # Convert to TTNN tensors
        self.cu_seqlens = ttnn.from_torch(
            cu_seqlens_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.cu_window_seqlens = ttnn.from_torch(
            cu_window_seqlens_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        cos, sin = position_embeddings
        cos, sin = convert_rope_style_hf_to_meta(cos, sin)
        cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - 14308), value=1).unsqueeze(0).unsqueeze(0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - 14308), value=0).unsqueeze(0).unsqueeze(0)
        self.rot_mats = [
            ttnn.from_torch(
                cos,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            ),
            ttnn.from_torch(
                sin,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            ),
        ]
        self.patch_input = self.reference_model.patch_embed(self.pt_pixel_values)
        self.tt_patch_input = self.tt_model.prepare_input(self.patch_input, self.window_index)

        self.reference_model.eval()  # Ensure eval mode

        # Make sure pt_pixel_values and image_grid_thw are on CPU and correct dtype
        pt_pixel_values = self.pt_pixel_values.float().cpu()
        image_grid_thw = self.image_grid_thw.cpu()

        # Generate reference output for validation
        with torch.no_grad():
            self.torch_output = self.reference_model(pt_pixel_values, image_grid_thw)

    def setup_l1_sharded_input(self, input_tensor=None):
        # For Qwen, just return the patch_input for now
        return self.patch_input

    def setup_dram_sharded_input(self, device):
        # For Qwen, just return the patch_input and other configs
        input_mem_config = ttnn.DRAM_MEMORY_CONFIG
        return (
            self.patch_input,
            self.tt_patch_input,
            input_mem_config,
            self.window_index,
            self.cu_seqlens,
            self.cu_window_seqlens,
            self.rot_mats,
        )

    def get_host_tensor(self):
        # Get the exact shape from the device tensor
        device_shape = self.tt_patch_input.shape

        # Convert TTNN shape to tuple for PyTorch
        shape_tuple = tuple(device_shape)

        # Create a PyTorch tensor with the exact same shape
        # Fill with zeros and then copy the actual data where it belongs
        host_tensor_torch = torch.zeros(shape_tuple, dtype=torch.float32)

        # Get the original patch input
        patch_input_cpu = self.patch_input

        # Copy the actual data into the right position (assuming batch=0, first sequence)
        actual_seq_len = patch_input_cpu.shape[0]
        host_tensor_torch[0, 0, :actual_seq_len, :] = patch_input_cpu[:actual_seq_len, :]

        # Convert to TTNN host tensor with matching dtype and layout
        return ttnn.from_torch(
            host_tensor_torch,
            dtype=self.act_dtype,  # Match the device tensor dtype
            layout=ttnn.TILE_LAYOUT,  # Match the device tensor layout
        )

    def run(self):
        self.ttnn_output_tensor = self.tt_model(
            self.ttnn_input,  # Use the TTNN tensor, not the PyTorch tensor
            unpadded_seq_len=14308,
            cu_seqlens=self.cu_seqlens,
            cu_window_seqlens=self.cu_window_seqlens,
            rot_mats=self.rot_mats,
        )

    def validate(self, output_tensor=None):
        tt_out = self.ttnn_output_tensor
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.device, dims=(1, 3), mesh_shape=self.model_args.cluster_shape),
        )
        tt_output_torch = (
            tt_out[:, 0:1, :, : self.model_args.hf_config.vision_config.out_hidden_size].squeeze(0).squeeze(0)
        )
        tt_output_torch = tt_output_torch[torch.argsort(self.window_index), :]
        passing, pcc_message = comp_pcc(self.torch_output, tt_output_torch, self.valid_pcc)
        logger.info(f"Qwen25VL PCC: {pcc_message}")
        # Enable validation now that we're not in trace capture
        assert passing, f"PCC value is lower than {self.valid_pcc} for some of the outputs. Check Warnings!"

    def dealloc_output(self):
        if hasattr(self, "ttnn_output_tensor") and self.ttnn_output_tensor is not None:
            ttnn.deallocate(self.ttnn_output_tensor)
