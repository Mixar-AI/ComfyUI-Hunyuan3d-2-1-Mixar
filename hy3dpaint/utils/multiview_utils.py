# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
from ..hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

# Global cache for multiviewDiffusionNet instances
_mvd_cache: dict = {}

def get_mvd_cache_key(config, paint_model=None):
    """Generate a cache key from config + paint_model identity."""
    return (config.multiview_cfg_path, config.scheduler, config.device, id(paint_model))

def clear_mvd_cache():
    """Clear the multiview diffusion net cache."""
    global _mvd_cache
    _mvd_cache.clear()


class multiviewDiffusionNet:
    def __init__(self, config, paint_model=None) -> None:
        global _mvd_cache

        # Check cache for existing instance
        cache_key = get_mvd_cache_key(config, paint_model)
        if cache_key in _mvd_cache:
            cached = _mvd_cache[cache_key]
            print("[multiviewDiffusionNet] Cache hit — reusing existing instance")
            self.device = cached.device
            self.cfg = cached.cfg
            self.mode = cached.mode
            self.pipeline = cached.pipeline
            if hasattr(cached, 'dino_v2'):
                self.dino_v2 = cached.dino_v2
            return

        print("[multiviewDiffusionNet] Cache miss — creating new instance")
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        if paint_model is not None:
            # Use pre-loaded models from the loader node
            pipeline = paint_model["pipeline"]

            if config.scheduler == "UniPCMultistep":
                pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            elif config.scheduler == "DDIM":
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            else:
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
            pipeline.set_progress_bar_config(disable=False)
            pipeline.eval()
            setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
            self.pipeline = pipeline.to(self.device)
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()

            if paint_model.get("dino_v2") is not None:
                self.dino_v2 = paint_model["dino_v2"].to(torch.float16).to(self.device)
            elif hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
                from ..hunyuanpaintpbr.unet.modules import Dino_v2
                dino_path = getattr(config, 'dino_ckpt_path', 'facebook/dinov2-giant')
                self.dino_v2 = Dino_v2(dino_path).to(torch.float16).to(self.device)
        else:
            # Fallback: download from HuggingFace Hub (for direct Python usage)
            import huggingface_hub

            pretrained_path = getattr(config, 'multiview_pretrained_path', 'tencent/Hunyuan3D-2.1')
            model_path = huggingface_hub.snapshot_download(
                repo_id=pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
            )

            model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")

            pipeline = HunyuanPaintPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )

            if config.scheduler == "UniPCMultistep":
                pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            elif config.scheduler == "DDIM":
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            else:
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
            pipeline.set_progress_bar_config(disable=False)
            pipeline.eval()
            setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
            self.pipeline = pipeline.to(self.device)
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()

            if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
                from ..hunyuanpaintpbr.unet.modules import Dino_v2
                dino_path = getattr(config, 'dino_ckpt_path', 'facebook/dinov2-giant')
                self.dino_v2 = Dino_v2(dino_path).to(torch.float16).to(self.device)

        # Store in cache
        _mvd_cache[cache_key] = self

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0, camera_azims=None, guidance_rescale=0.0):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input, num_steps=num_steps, guidance_scale=guidance_scale, seed=seed, camera_azims=camera_azims, guidance_rescale=guidance_rescale
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0, camera_azims=None, guidance_rescale=0.0):
        self.seed_everything(seed)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        
        if not isinstance(input_images, List):
            input_images = [input_images]
            
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
            
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if camera_azims is not None:
            kwargs["camera_azims"] = camera_azims

        kwargs["guidance_rescale"] = guidance_rescale

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 10,
            "UniPCMultistepScheduler": 10,
            "DDIMScheduler": 10,
            "ShiftSNRScheduler": 10,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_steps,
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
