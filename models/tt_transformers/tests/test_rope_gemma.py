### Required environment variables for launch.json:
# "MESH_DEVICE": "T3K"
# "TT_METAL_HOME": "${workspaceFolder}",
# "PYTHON_ENV_DIR": "${workspaceFolder}/python_env",
# "PYTHONPATH": "${workspaceFolder}",
# "HF_MODEL": "google/gemma-3-27b-it",
# "HF_HOME": "/localdev/huggingface"
# "TT_CACHE_PATH": "/localdev/huggingface/tt_cache/google/gemma-3-27b-it",


import os

import matplotlib.pyplot as plt
import pytest
import torch
from transformers import Gemma3ForConditionalGeneration

import ttnn
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup


def extract_cos_sin(rotary_emb, max_pos=100001):
    """
    Extract cos and sin tensors from a rotary embedding module for positions 0 to max_pos-1.
    Returns cos, sin of shape [max_pos, head_dim].
    """
    head_dim = rotary_emb.config.head_dim
    positions = torch.arange(max_pos, dtype=torch.int64)
    position_ids = positions.unsqueeze(0)  # [1, max_pos]
    dummy_x = torch.zeros(1, max_pos, head_dim)
    with torch.no_grad():
        cos, sin = rotary_emb(dummy_x, position_ids)  # [1, max_pos, head_dim] each
    return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def plot_cos_sin_frequencies(ref, tt, out_prefix="global_rope"):
    """
    Plot sin and cos for the lowest and highest frequency (first and last dim).
    Save to PNG files with enhanced visibility for overlapping lines.
    """
    max_pos, head_dim = ref.shape

    # Convert to numpy for easier handling
    ref_np = ref.cpu().numpy()
    tt_np = tt.cpu().numpy()

    # Lowest frequency: dim 0 (show only first 100 positions for clarity)
    num_samples = 100
    plt.figure(figsize=(12, 6))

    # Plot with different styles to make both lines visible
    positions = range(num_samples)
    plt.plot(
        positions,
        ref_np[:num_samples, 0],
        "o-",
        color="red",
        linewidth=2,
        markersize=4,
        alpha=0.8,
        label="Reference (Gemma3)",
    )
    plt.plot(
        positions,
        tt_np[:num_samples, 0],
        "s--",
        color="blue",
        linewidth=2,
        markersize=3,
        alpha=0.8,
        label="TT Transformers",
    )

    # Add difference plot if values are very close
    diff = abs(ref_np[:num_samples, 0] - tt_np[:num_samples, 0])
    max_diff = max(diff)
    if max_diff < 1e-6:
        plt.text(
            0.02,
            0.98,
            f"Max difference: {max_diff:.2e}\n(Lines may overlap)",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

    plt.title(f"{out_prefix}: Lowest Frequency (dim 0, first {num_samples} positions)")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_lowest_freq.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Highest frequency: dim head_dim-1 (sample every 1000 positions for visibility)
    sample_step = max(1, max_pos // 1000)  # Sample ~1000 points
    sample_positions = range(0, max_pos, sample_step)

    plt.figure(figsize=(12, 6))
    plt.plot(
        sample_positions,
        ref_np[::sample_step, -1],
        "o-",
        color="red",
        linewidth=1.5,
        markersize=2,
        alpha=0.8,
        label="Reference (Gemma3)",
    )
    plt.plot(
        sample_positions,
        tt_np[::sample_step, -1],
        "s--",
        color="blue",
        linewidth=1.5,
        markersize=1.5,
        alpha=0.8,
        label="TT Transformers",
    )

    # Add difference info for high frequency
    diff_high = abs(ref_np[:, -1] - tt_np[:, -1])
    max_diff_high = max(diff_high)
    if max_diff_high < 1e-6:
        plt.text(
            0.02,
            0.98,
            f"Max difference: {max_diff_high:.2e}\n(Lines may overlap)",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

    plt.title(f"{out_prefix}: Highest Frequency (dim {head_dim-1}, sampled every {sample_step} positions)")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_highest_freq.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create a difference plot to show where implementations diverge
    plt.figure(figsize=(12, 8))

    # Low frequency difference
    plt.subplot(2, 1, 1)
    diff_low = abs(ref_np[:num_samples, 0] - tt_np[:num_samples, 0])
    plt.semilogy(positions, diff_low + 1e-16, "g-", linewidth=2, marker="o", markersize=3)
    plt.title(f"{out_prefix}: Absolute Difference - Low Frequency (dim 0)")
    plt.xlabel("Position")
    plt.ylabel("Absolute Difference (log scale)")
    plt.grid(True, alpha=0.3)

    # High frequency difference
    plt.subplot(2, 1, 2)
    diff_high_sampled = abs(ref_np[::sample_step, -1] - tt_np[::sample_step, -1])
    plt.semilogy(sample_positions, diff_high_sampled + 1e-16, "g-", linewidth=2, marker="s", markersize=2)
    plt.title(f"{out_prefix}: Absolute Difference - High Frequency (dim {head_dim-1})")
    plt.xlabel("Position")
    plt.ylabel("Absolute Difference (log scale)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_differences.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print numerical summary
    print(f"\n📊 {out_prefix} Analysis:")
    print(f"  • Low freq (dim 0) max difference: {max(diff):.2e}")
    print(f"  • High freq (dim {head_dim-1}) max difference: {max_diff_high:.2e}")
    print(f"  • Low freq (dim 0) mean difference: {diff.mean():.2e}")
    print(f"  • High freq (dim {head_dim-1}) mean difference: {diff_high.mean():.2e}")

    if max(diff) < 1e-10 and max_diff_high < 1e-10:
        print(f"  ✅ Implementations are virtually identical!")
    elif max(diff) < 1e-6 and max_diff_high < 1e-6:
        print(f"  ✅ Implementations are very close (differences < 1e-6)")
    else:
        print(f"  ⚠️  Noticeable differences detected")


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128 * 1024,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_gemma3_rope_reference(seq_len, batch_size, mesh_device):
    model_id = os.environ.get("HF_MODEL", "google/gemma-3-27b-it")
    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()

    rotary_emb = model.model.language_model.rotary_emb
    rotary_emb_local = model.model.language_model.rotary_emb_local

    ref_cos, ref_sin = extract_cos_sin(rotary_emb, max_pos=seq_len)
    ref_cos_local, ref_sin_local = extract_cos_sin(rotary_emb_local, max_pos=seq_len)

    tt_model_args = ModelArgs(mesh_device, max_batch_size=1)

    rope_setup = RotarySetup(
        mesh_device,
        tt_model_args.max_batch_size,
        tt_model_args.head_dim,
        seq_len,
        tt_model_args.rope_theta,
        tt_model_args.rope_scaling,
    )

    rope_local_setup = RotarySetup(
        mesh_device,
        tt_model_args.max_batch_size,
        tt_model_args.head_dim,
        seq_len,
        tt_model_args.rope_theta_local,
        None,
    )

    # tt_cos = ttnn.to_torch(rope_setup.cos_matrix, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    # tt_sin = ttnn.to_torch(rope_setup.sin_matrix, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    # tt_cos_local = ttnn.to_torch(rope_local_setup.cos_matrix, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    # tt_sin_local = ttnn.to_torch(rope_local_setup.sin_matrix, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Remove first dimension
    # tt_cos = tt_cos.squeeze(0).float()
    # tt_sin = tt_sin.squeeze(0).float()
    # tt_cos_local = tt_cos_local.squeeze(0).float()
    # tt_sin_local = tt_sin_local.squeeze(0).float()

    # Prepare dummy Q and K inputs
    batch = 1
    n_heads = tt_model_args.n_heads
    n_kv_heads = tt_model_args.n_kv_heads
    head_dim = tt_model_args.head_dim

    input = torch.rand(batch, n_heads, seq_len, head_dim) * 2 - 1

    ref_out = apply_rotary_pos_emb(input, ref_cos, ref_sin)

    print(f"input shape: {input.shape}")
    print(f"n_heads: {tt_model_args.n_heads}")
    print(f"n_kv_heads: {tt_model_args.n_kv_heads}")
    print(f"head_dim: {tt_model_args.head_dim}")
    print(f"seq_len: {seq_len}")
    print(f"cluster_shape: {tt_model_args.cluster_shape}")

    ### THIS IS WHERE I'M STRUGGLING ###

    print(f"input shape: {input.shape}")
    inp_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=ttnn.num_cores_to_corerangeset(batch, rope_setup.core_grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(
        input, device=mesh_device, dtype=ttnn.bfloat16, memory_config=inp_mem_config, layout=ttnn.TILE_LAYOUT
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    print(f"tt_input shape: {tt_input.shape}")
    print(f"rope_setup.cos_matrix shape: {rope_setup.cos_matrix.shape}")
    print(f"rope_setup.sin_matrix shape: {rope_setup.sin_matrix.shape}")
    print(f"rope_setup.transformation_mat shape: {rope_setup.transformation_mat.shape}")

    tt_out = ttnn.experimental.rotary_embedding_llama(
        tt_input,
        rope_setup.cos_matrix,
        rope_setup.sin_matrix,
        rope_setup.transformation_mat,
        is_decode_mode=True,
        # compute_kernel_config=self.compute_kernel_config,
    )

    # Compare tt_out and ref_out
    tt_out_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    # Ensure shapes match
    assert tt_out_torch.shape == ref_out.shape, f"Shape mismatch: {tt_out_torch.shape} vs {ref_out.shape}"
    # Compute absolute and relative error
    abs_diff = (tt_out_torch - ref_out).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    rel_diff = abs_diff / (ref_out.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    print(f"Max absolute difference: {max_abs_diff:.3e}")
    print(f"Mean absolute difference: {mean_abs_diff:.3e}")
    print(f"Max relative difference: {max_rel_diff:.3e}")
    print(f"Mean relative difference: {mean_rel_diff:.3e}")
    # Optionally, assert that the difference is within a reasonable tolerance
    assert max_abs_diff < 1e-3, f"Max absolute difference too high: {max_abs_diff}"

    # plot_cos_sin_frequencies(ref_cos, tt_cos, out_prefix="gemma3_cos_global")
    # plot_cos_sin_frequencies(ref_cos_local, tt_cos_local, out_prefix="gemma3_cos_local")
    # plot_cos_sin_frequencies(ref_sin, tt_sin, out_prefix="gemma3_sin_global")
    # plot_cos_sin_frequencies(ref_sin_local, tt_sin_local, out_prefix="gemma3_sin_local")
