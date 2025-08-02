import torch
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc, logger

from mmcv.utils import ext_loader
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch import nn
from typing import Optional

# Load external C++/CUDA module for modulated deform conv
ext_module = ext_loader.load_ext("_ext", ["modulated_deform_conv_forward"])


class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
    ) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(f"Expected 4D input, got {input.dim()}D tensor instead.")

        # Save convolution params
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None

        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor

        # Type consistency
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore

        # Save for backward
        ctx.save_for_backward(input, offset, mask, weight, bias)

        # Prepare output tensor and buffers
        output_size = ModulatedDeformConv2dFunction._output_size(ctx, input, weight)
        output = input.new_empty(output_size)
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]

        # Call C++/CUDA extension
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )

        return output

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be " + "x".join(map(str, output_size)) + ")")
        return output_size


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_modulated_deform_conv(device, reset_seeds):
    modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply

    # Load inputs
    x = torch.load("x.pt")
    x_tt = torch.load("x_tt.pt")
    _, msg = assert_with_pcc(x, x_tt, 0.99)
    logger.info(msg)

    # Load offsets
    offset = torch.load("offset.pt")
    offset_tt = torch.load("offset_tt.pt")
    _, msg = assert_with_pcc(offset, offset_tt, 0.99)
    logger.info(msg)

    # Load masks
    mask = torch.load("mask.pt")
    mask_tt = torch.load("mask_tt.pt")
    _, msg = assert_with_pcc(mask, mask_tt, 0.99)
    logger.info(msg)

    # Load weights and bias
    weight = torch.load("weight.pt")
    bias = torch.load("bias.pt")

    # Torch output
    torch_out = modulated_deform_conv2d(
        x,
        offset,
        mask,
        weight,
        bias,
        (1, 1),  # stride
        (1, 1),  # padding
        (1, 1),  # dilation
        1,  # groups
        1,  # deform_groups
    )

    # TT output
    tt_out = modulated_deform_conv2d(x_tt, offset_tt, mask_tt, weight, bias, (1, 1), (1, 1), (1, 1), 1, 1)

    # Compare outputs
    _, msg = assert_with_pcc(torch_out, tt_out, 0.99)
    logger.info(msg)
