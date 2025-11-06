# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

from ...encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ...models.transformers.transformer_qwenimage import QwenImageTransformer
from ...models.vae.vae_sd35 import VAEDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor


class QwenImagePipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        use_torch_text_encoder: bool = False,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
    ) -> None:
        self.timing_collector = None

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width

        # setup encoder and vae parallel configs.
        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        self._encoder_parallel_config = encoder_parallel_config
        self._vae_parallel_config = vae_parallel_config

        # Create submeshes based on CFG parallel configuration
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[
            0 : parallel_config.cfg_parallel.factor
        ]
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_device = self._submesh_devices[0]
        self.encoder_mesh_shape = ttnn.MeshShape(1, self._encoder_parallel_config.tensor_parallel.factor)
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = 0  # Use submesh 0 for VAE

        logger.info("loading models...")

        checkpoint_name = "Qwen/Qwen-Image"

        torch_transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        torch_transformer.eval()

        self._torch_vae = AutoencoderKLQwenImage.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLQwenImage)

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")

        logger.info("creating TT-NN transformer...")

        head_dim = torch_transformer.config.attention_head_dim
        num_heads = torch_transformer.config.num_attention_heads
        self._num_channels_latents = 16  # TODO: correct?
        self._patch_size = 2
        self._vae_scale_factor = 8  # TODO: correct?

        if num_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                num_heads,
                head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = QwenImageTransformer(
                patch_size=torch_transformer.config.patch_size,
                in_channels=torch_transformer.in_channels,
                num_layers=torch_transformer.config.num_layers,
                attention_head_dim=head_dim,
                num_attention_heads=num_heads,
                joint_attention_dim=torch_transformer.config.joint_attention_dim,
                out_channels=torch_transformer.out_channels,
                device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
            )

            if not cache.initialize_from_cache(
                tt_model=tt_transformer,
                torch_model=torch_transformer,
                model_name="qwen-image",
                subfolder="transformer",
                parallel_config=self._parallel_config,
                mesh_shape=tuple(submesh_device.shape),
                dtype="bf16",
            ):
                logger.info("Loading transformer weights from PyTorch state dict")
                tt_transformer.load_torch_state_dict(torch_transformer.state_dict())

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._latents_scaling = 1.0 / torch.tensor(self._torch_vae.config.latents_std).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )
        self._latents_shift = torch.tensor(self._torch_vae.config.latents_mean).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )

        # TDOD: in the reference, the factor is multiplied by 2
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        with self.encoder_reshape(self.encoder_device):
            logger.info("creating TT-NN text encoder...")
            self._text_encoder = Qwen25VlTokenizerEncoderPair(
                checkpoint_name,
                max_batch_size=2,
                max_sequence_length=1024,
                device=self.encoder_device,
                use_torch=use_torch_text_encoder,
            )

            ttnn.synchronize_device(self.encoder_device)

            logger.info("creating TT-NN VAE decoder...")
            self._vae_decoder = VAEDecoder.from_torch(
                torch_ref=self._torch_vae.decoder,
                mesh_device=self.vae_device,
                parallel_config=self._vae_parallel_config,
                ccl_manager=self._ccl_managers[self.vae_submesh_idx],
            )
            ttnn.synchronize_device(self.encoder_device)

        self._traces = None
        self.warmup()

    @contextmanager
    def encoder_reshape(self, device: ttnn.MeshDevice):
        original_mesh_shape = ttnn.MeshShape(tuple(device.shape))
        assert original_mesh_shape.mesh_size() == self.encoder_mesh_shape.mesh_size(), (
            f"Device cannot be reshaped device shape: {device.shape} encoder mesh shape: {self.encoder_mesh_shape}"
        )
        if original_mesh_shape != self.encoder_mesh_shape:
            device.reshape(self.encoder_mesh_shape)
        yield
        if original_mesh_shape != device.shape:
            device.reshape(original_mesh_shape)

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        dit_cfg: tuple[int, int] | None = None,
        dit_sp: tuple[int, int] | None = None,
        dit_tp: tuple[int, int] | None = None,
        encoder_tp: tuple[int, int] | None = None,
        vae_tp: tuple[int, int] | None = None,
        use_torch_text_encoder: bool = False,
        num_links: int,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
    ) -> QwenImagePipeline:
        default_config = {
            (2, 4): {
                "cfg_config": (2, 1),
                "sp": (2, 0),
                "tp": (2, 1),
                "encoder_tp": (4, 1),
                "vae_tp": (4, 1),
                "num_links": 1,
            },
            (4, 8): {
                "cfg_config": (2, 1),
                "sp": (4, 0),
                "tp": (4, 1),
                "encoder_tp": (4, 1),
                "vae_tp": (4, 1),
                "num_links": 4,
            },
        }
        cfg_factor, cfg_axis = dit_cfg or default_config[tuple(mesh_device.shape)]["cfg_config"]
        sp_factor, sp_axis = dit_sp or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = dit_tp or default_config[tuple(mesh_device.shape)]["tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or default_config[tuple(mesh_device.shape)]["encoder_tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or default_config[tuple(mesh_device.shape)]["vae_tp"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {dit_parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel config: {vae_parallel_config}")

        pipeline = QwenImagePipeline(
            mesh_device=mesh_device,
            use_torch_text_encoder=use_torch_text_encoder,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
        )

        return pipeline

    def warmup(self):
        """Warmup the pipeline by running a single inference."""
        logger.info("Pipeline warmup...")
        self.run_single_prompt(
            prompt="", negative_prompt=None, num_inference_steps=2, cfg_scale=3.5, seed=0, traced=False
        )
        self._sync_devices()

    def run_single_prompt(
        self, prompt, negative_prompt=None, num_inference_steps=40, cfg_scale=5.0, seed=None, traced=True
    ):
        return self.__call__(
            prompts=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompts=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            traced=traced,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
    ) -> list[Image.Image]:
        timer = self.timing_collector.reset() if self.timing_collector else None
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        spatial_sequence_length = latents_height * latents_width

        with timer.time_section("total") if timer else nullcontext():
            cfg_enabled = cfg_scale > 1
            logger.info("encoding prompts...")

            with timer.time_section("total_encoding") if timer else nullcontext():
                with self.encoder_reshape(self.encoder_device):
                    prompt_embeds1, pooled_prompt_embeds1, prompt_embeds2, pooled_prompt_embeds2 = self._encode_prompts(
                        prompts=prompts,
                        negative_prompts=negative_prompts,
                        num_images_per_prompt=num_images_per_prompt,
                        cfg_enabled=cfg_enabled,
                    )

            logger.info("preparing timesteps...")
            timesteps, sigmas = _schedule(
                self._scheduler,
                step_count=num_inference_steps,
                spatial_sequence_length=spatial_sequence_length,
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [
                prompt_count * num_images_per_prompt,
                self._num_channels_latents,
                self._height // self._vae_scale_factor,
                self._width // self._vae_scale_factor,
            ]
            # We let randn generate a permuted latent tensor in float32, so that the generated noise
            # matches the reference implementation.
            latents = self.transformers[0].patchify(
                torch.randn(shape, dtype=torch.float32).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)
            )

            tt_prompt_embeds_device_list = []
            tt_prompt_embeds1_list = []
            tt_prompt_embeds2_list = []
            tt_pooled_prompt_embeds_device_list = []
            tt_pooled_prompt_embeds1_list = []
            tt_pooled_prompt_embeds2_list = []
            tt_latents_step_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds_device = tensor.from_torch(
                    prompt_embeds1[i : i + 1] if cfg_factor == 2 else prompt_embeds1,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_prompt_embeds1 = tensor.from_torch(
                    prompt_embeds1[i : i + 1] if cfg_factor == 2 else prompt_embeds1,
                    device=submesh_device,
                    on_host=True,
                )
                tt_prompt_embeds2 = tensor.from_torch(
                    prompt_embeds2[i : i + 1] if cfg_factor == 2 else prompt_embeds2,
                    device=submesh_device,
                    on_host=True,
                )

                tt_pooled_prompt_embeds_device = tensor.from_torch(
                    pooled_prompt_embeds1[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds1,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_pooled_prompt_embeds1 = tensor.from_torch(
                    pooled_prompt_embeds1[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds1,
                    device=submesh_device,
                    on_host=True,
                )
                tt_pooled_prompt_embeds2 = tensor.from_torch(
                    pooled_prompt_embeds2[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds2,
                    device=submesh_device,
                    on_host=True,
                )

                tt_initial_latents = tensor.from_torch(
                    latents, device=submesh_device, on_host=traced, mesh_axes=[None, sp_axis, None]
                )

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds_device = tt_prompt_embeds_device.to(submesh_device)
                        tt_pooled_prompt_embeds_device = tt_pooled_prompt_embeds_device.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds_device, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds_device, self._traces[i].pooled_input)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds_device = self._traces[i].prompt_input
                        tt_pooled_prompt_embeds_device = self._traces[i].pooled_input

                tt_prompt_embeds_device_list.append(tt_prompt_embeds_device)
                tt_prompt_embeds1_list.append(tt_prompt_embeds1)
                tt_prompt_embeds2_list.append(tt_prompt_embeds2)
                tt_pooled_prompt_embeds_device_list.append(tt_pooled_prompt_embeds_device)
                tt_pooled_prompt_embeds1_list.append(tt_pooled_prompt_embeds1)
                tt_pooled_prompt_embeds2_list.append(tt_pooled_prompt_embeds2)
                tt_latents_step_list.append(tt_initial_latents)

            logger.info("denoising...")

            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with timer.time_step("denoising_step") if timer else nullcontext():
                    sigma_difference = sigmas[i + 1] - sigmas[i]

                    tt_timestep_list = []
                    tt_sigma_difference_list = []
                    for submesh_nr, submesh_device in enumerate(self._submesh_devices):
                        tt_timestep = ttnn.full(
                            [1, 1],
                            fill_value=t,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.float32,
                            device=submesh_device if not traced else None,
                        )
                        tt_timestep_list.append(tt_timestep)

                        tt_sigma_difference = ttnn.full(
                            [1, 1],
                            fill_value=sigma_difference,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            device=submesh_device if not traced else None,
                        )
                        tt_sigma_difference_list.append(tt_sigma_difference)

                        if t >= 1000 * negative_strategy_switch_time:
                            ttnn.copy_host_to_device_tensor(
                                tt_prompt_embeds1_list[submesh_nr],
                                tt_prompt_embeds_device_list[submesh_nr],
                            )
                            ttnn.copy_host_to_device_tensor(
                                tt_pooled_prompt_embeds1_list[submesh_nr],
                                tt_pooled_prompt_embeds_device_list[submesh_nr],
                            )
                        else:
                            ttnn.copy_host_to_device_tensor(
                                tt_prompt_embeds2_list[submesh_nr],
                                tt_prompt_embeds_device_list[submesh_nr],
                            )
                            ttnn.copy_host_to_device_tensor(
                                tt_pooled_prompt_embeds2_list[submesh_nr],
                                tt_pooled_prompt_embeds_device_list[submesh_nr],
                            )

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        latents=tt_latents_step_list,
                        cfg_enabled=cfg_enabled,
                        prompt_embeds=tt_prompt_embeds_device_list,
                        pooled_prompt_embeds=tt_pooled_prompt_embeds_device_list,
                        cfg_scale=cfg_scale,
                        sigma_difference=tt_sigma_difference_list,
                        traced=traced,
                    )

            logger.info("decoding image...")

            with timer.time_section("vae_decoding") if timer else nullcontext():
                # Sync because we don't pass a persistent buffer or a barrier semaphore.
                ttnn.synchronize_device(self.vae_device)

                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents_step_list[self.vae_submesh_idx],
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )

                torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])

                torch_latents = self.transformers[0].unpatchify(
                    torch_latents,
                    height=latents_height,
                    width=latents_width,
                )

                torch_latents = torch_latents / self._latents_scaling + self._latents_shift

                with self.encoder_reshape(self.encoder_device):
                    tt_latents = tensor.from_torch(torch_latents, device=self.vae_device)
                    tt_decoded_output = self._vae_decoder(tt_latents)
                    decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_index: int,
    ) -> ttnn.Tensor:
        if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1:
            latent = ttnn.concat([latent, latent])

        return self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            pooled=pooled,
            timestep=timestep,
        )

    def _sync_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor
        timestep: list[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: list[ttnn.Tensor],  # device tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
        traced: bool,
    ) -> list[ttnn.Tensor]:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                # pred = self._step_inner(
                #     cfg_enabled=cfg_enabled,
                #     latent=latents[submesh_id],
                #     prompt=prompt_embeds[submesh_id],
                #     pooled=pooled_prompt_embeds[submesh_id],
                #     timestep=timestep_device,
                #     submesh_index=submesh_id,
                # )

                # for device in self._submesh_devices:
                #     ttnn.synchronize_device(device)

                # Already cached during warmup. Just capture.
                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        pooled_input=pooled_prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        latents_output=pred,
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)

            # TODO: If we don't do this, we get noise when tracing is enabled. But why, since sigma
            # difference is only used outside of tracing region?
            sigma_difference_device = [trace.sigma_difference_input for trace in self._traces]
        else:
            for submesh_id in range(len(self._submesh_devices)):
                noise_pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep[submesh_id],
                    submesh_index=submesh_id,
                )
                noise_pred_list.append(noise_pred)

            sigma_difference_device = sigma_difference

        if cfg_enabled:
            if self._parallel_config.cfg_parallel.factor == 1:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + cfg_scale * (cond - uncond)
            else:
                # uncond and cond are replicated, so it is fine to get a single tensor from each
                uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True)).to(
                    torch.float32
                )
                cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True)).to(
                    torch.float32
                )

                torch_noise_pred = uncond + cfg_scale * (cond - uncond)

                noise_pred_list[0] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[0], mesh_axes=[None, sp_axis, None]
                )

                noise_pred_list[1] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[1], mesh_axes=[None, sp_axis, None]
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
            ttnn.multiply_(noise_pred_list[submesh_id], sigma_difference_device[submesh_id])
            ttnn.add_(latents[submesh_id], noise_pred_list[submesh_id])

        return latents

    def _encode_prompts(
        self,
        *,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_images_per_prompt: int,
        cfg_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        timer = self.timing_collector

        no_negative_prompt = [x is None for x in negative_prompts]
        negative_prompts = [x if x is not None else "" for x in negative_prompts]

        with timer.time_section("text_encoding") if timer else nullcontext():
            pos_prompt_embeds, pos_pooled_prompt_embeds = self._text_encoder.encode(
                prompts,
                num_images_per_prompt=num_images_per_prompt,
            )

            neg_prompt_embeds, neg_pooled_prompt_embeds = self._text_encoder.encode(
                negative_prompts,
                num_images_per_prompt=num_images_per_prompt,
            )

        if not cfg_enabled:
            return pos_prompt_embeds, pos_pooled_prompt_embeds, pos_prompt_embeds, pos_pooled_prompt_embeds

        zeroed_prompt_embeds = neg_prompt_embeds.clone()
        zeroed_pooled_prompt_embeds = neg_pooled_prompt_embeds.clone()

        for i, no_neg in enumerate(no_negative_prompt):
            if no_neg:
                zeroed_prompt_embeds[i] = 0
                zeroed_pooled_prompt_embeds[i] = 0

        prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0)
        prompt_embeds_alt = torch.cat([zeroed_prompt_embeds, pos_prompt_embeds], dim=0)

        pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)
        pooled_prompt_embeds_alt = torch.cat([zeroed_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds, prompt_embeds_alt, pooled_prompt_embeds_alt


def _schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    step_count: int,
    spatial_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / step_count, step_count),
        mu=_calculate_shift(
            spatial_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        ),
    )

    return scheduler.timesteps, scheduler.sigmas


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
