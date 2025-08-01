# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.uniad.tt.tt_fpn import TtFPN
from models.experimental.uniad.tt.ttnn_resnet import TtResNet


class TtDetection:
    def __init__(
        self,
        device,
        parameters,
    ):
        self.device = device
        self.img_backbone = TtResNet(
            parameters.conv_args["img_backbone"],
            parameters["img_backbone"]["res_model"],
            device,
            depth=101,
            in_channels=3,
            stem_channels=None,
            base_channels=64,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(1, 2, 3),
            style="caffe",
            deep_stem=False,
            avg_down=False,
            frozen_stages=4,
            conv_cfg=None,
            dcn=True,
            stage_with_dcn=(False, False, True, True),
            # zero_init_residual=True,
            pretrained=None,
            init_cfg=None,
        )

        self.img_neck = TtFPN(
            parameters["img_neck"]["model_args"],
            parameters["img_neck"],
            device=self.device,
        )

    def __call__(self, x):
        x = self.img_backbone(x)
        x = self.img_neck(x)
        return x
