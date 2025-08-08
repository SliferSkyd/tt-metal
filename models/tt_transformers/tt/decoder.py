# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class TransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.state_dict_prefix = args.get_state_dict_prefix(self.__class__.__name__, layer_num)

        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention

        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )

        def pre_attention_norm(weight_key):
            return DistributedNorm(
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    state_dict=self.state_dict,
                    state_dict_prefix=None,
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key=weight_key,
                    is_distributed=self.args.is_distributed_norm,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                ),
                args,
                TG=args.is_galaxy,
            )

        def post_attention_norm(weight_key):
            return DistributedNorm(
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    state_dict=state_dict,
                    state_dict_prefix=None,
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key=weight_key,
                    is_distributed=self.args.is_distributed_norm,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ),
                args,
                TG=args.is_galaxy,
            )

        def norm_and_fracture(x, norm, mode):
            skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
            normed = norm(x, mode)
            reduced = tt_all_reduce(
                normed,
                self.mesh_device,
                cluster_axis=0,
                dim=0 if (self.args.is_galaxy and self.dim < 8192) else 3,
                num_reduce_scatter_links=self.args.num_reduce_scatter_links,
                num_all_gather_links=self.args.num_all_gather_links,
                topology=self.args.ccl_topology(),
                memory_config=skip_mem_cfg,
                sharded=True if mode == "decode" else False,
                use_composite=self.dim == 8192,
            )
            return reduced

        # Check state_dict to see if we need to use norms around feedforward
        pre_post_feedforward_norm_weight_keys = ["pre_feedforward_layernorm", "post_feedforward_layernorm"]
        self.use_all_norms = all(
            f"{self.state_dict_prefix}{norm_str}.weight" in self.state_dict
            for norm_str in pre_post_feedforward_norm_weight_keys
        )

        # Full flow:                            # Llama3 flow:
        # pre_attention_norm                    # pre_attention_norm
        # attention                             # attention
        # post_attention_norm                   # ------------
        # add residual                          # add residual
        # pre_ff_norm                           # pre_ff_norm (uses post_attention_norm weight)
        # feed_forward                          # feed_forward
        # post_ff_norm                          # ------------
        # add residual                          # add residual

        pre_attention_norm_str = f"{self.state_dict_prefix}attention_norm"
        post_attention_norm_str = f"{self.state_dict_prefix}ffn_norm" if self.use_all_norms else None
        pre_ff_norm_str = (
            f"{self.state_dict_prefix}pre_feedforward_layernorm"
            if self.use_all_norms
            else f"{self.state_dict_prefix}ffn_norm"
        )
        post_ff_norm_str = f"{self.state_dict_prefix}post_feedforward_layernorm" if self.use_all_norms else None

        # Set up norms based on weights in state_dict
        if f"{pre_attention_norm_str}.weight" in self.state_dict:
            self.pre_attention_norm = pre_attention_norm(pre_attention_norm_str)
        else:
            # ttnn.Tensor(x) only increases Tensor's internal refcount, used in deallocate()
            self.pre_attention_norm = lambda x, mode: ttnn.Tensor(x)

        # Tensor is fractured across devices before this norm. Keep the flow by fracturing it after norm.
        if f"{post_attention_norm_str}.weight" in self.state_dict:
            # Call post_attention_norm here to initialize weights
            _post_attention_norm = post_attention_norm(post_attention_norm_str)
            self.post_attention_norm = lambda x, mode: norm_and_fracture(x, _post_attention_norm, mode)
        else:
            self.post_attention_norm = lambda x, mode: ttnn.Tensor(x)

        if f"{pre_ff_norm_str}.weight" in self.state_dict:
            self.pre_ff_norm = post_attention_norm(pre_ff_norm_str)
        else:
            self.pre_ff_norm = lambda x, mode: ttnn.Tensor(x)

        # Tensor is fractured across devices before this norm. Keep the flow by fracturing it after norm.
        if f"{post_ff_norm_str}.weight" in self.state_dict:
            # Call post_attention_norm here to initialize weights
            _post_ff_norm = post_attention_norm(post_ff_norm_str)
            self.post_ff_norm = lambda x, mode: norm_and_fracture(x, _post_ff_norm, mode)
        else:
            self.post_ff_norm = lambda x, mode: ttnn.Tensor(x)

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy

        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # Choose the correct rotation matrices based on the mode
        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # Norms take fractured inputs and output replicated across devices
        attn_in = self.pre_attention_norm(x, mode)

        # Attention takes replicated inputs and produces fractured outputs
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        attn_in.deallocate()
        # Here x and attn_out are both fractured across devices

        # If triggered, post_attention_norm will produce fractured outputs
        post_attn_out = self.post_attention_norm(attn_out, mode)
        attn_out.deallocate()

        add_out = ttnn.add(
            x,
            post_attn_out,
            memory_config=skip_mem_cfg,
            dtype=ttnn.bfloat16 if TG else None,
        )
        x.deallocate()
        post_attn_out.deallocate()

        # Norms take fractured inputs and output replicated across devices
        if TG and mode == "decode":
            add_out = ttnn.to_memory_config(add_out, memory_config=self.model_config["MLP_ACT_MEMCFG"])

        pre_ffn_out = self.pre_ff_norm(add_out, mode)

        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward(pre_ffn_out, mode)
        pre_ffn_out.deallocate()

        # If triggered, post_ff_norm will produce fractured outputs
        post_ffn_out = self.post_ff_norm(ff_out, mode)
        ff_out.deallocate()

        # post_ffn_out and add_out are both fractured across devices
        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )
        out = ttnn.add(
            add_out,
            post_ffn_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )
        add_out.deallocate()
        post_ffn_out.deallocate()

        return out  # fractured across devices
