"""LIME: the backbone and the flow head put together."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ..pose import PoseNorm, pose9_to_mat
from .backbone import Backbone
from .flow_head import FlowMatchingPoseHead, euler_integrate

logger = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CHECKPOINT_FILES = ("config.json", "flow_head.safetensors", "flow_head_config.json",
                    "pose_norm_stats.json")


@dataclass
class Prediction:
    """One prediction. Poses are [tx, ty, tz, r1..r6] in metres; see `lime_camera.pose`."""
    intent: str
    observation_gain: str
    pose9_mean: np.ndarray            # (9,)
    pose9_std: np.ndarray             # (9,)
    pose9_samples: np.ndarray         # (K, 9)
    timings: dict = field(default_factory=dict)

    @property
    def T_mean(self) -> np.ndarray:
        """Mean target pose as a 4x4 matrix: the end camera in the start-camera frame."""
        return pose9_to_mat(self.pose9_mean)

    @property
    def T_samples(self) -> np.ndarray:
        """Sampled target poses as (K, 4, 4) matrices."""
        return np.stack([pose9_to_mat(p) for p in self.pose9_samples])

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "observation_gain": self.observation_gain,
            "pose9_mean": self.pose9_mean.tolist(),
            "pose9_std": self.pose9_std.tolist(),
            "pose9_samples": self.pose9_samples.tolist(),
            "T_mean": self.T_mean.tolist(),
            "T_samples": self.T_samples.tolist(),
            "timings": self.timings,
        }


class LIME:
    """Predict where the camera should move next, from an image and an intent."""

    def __init__(self, backbone: Backbone, flow_head: FlowMatchingPoseHead,
                 pose_norm: PoseNorm):
        self.backbone = backbone
        self.flow_head = flow_head
        self.pose_norm = pose_norm

    @classmethod
    def from_pretrained(cls, checkpoint, dtype: str = "bf16",
                        attn_implementation: str = "sdpa") -> "LIME":
        """Load a checkpoint directory.

        dtype applies to the backbone: bf16 (default), fp16 or fp32.
        attn_implementation is sdpa (default, no extra deps) or flash_attention_2.
        """
        checkpoint = Path(checkpoint)
        missing = [name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{checkpoint} has no {', '.join(missing)}. Point at the folder that directly "
                f"contains config.json and the flow head. scripts/download_weights.sh "
                f"downloads the released checkpoint to weights/lime_v1.")
        backbone = Backbone.from_pretrained(checkpoint, DTYPES[dtype], attn_implementation)
        config = json.loads((checkpoint / "flow_head_config.json").read_text())
        flow_head = FlowMatchingPoseHead(**config).to(backbone.device).eval()
        flow_head.load_state_dict(load_file(checkpoint / "flow_head.safetensors"), strict=True)
        logger.info("loaded the flow head: predict_x=%s", flow_head.predict_x)
        return cls(backbone, flow_head, PoseNorm.from_file(checkpoint / "pose_norm_stats.json"))

    def predict(self, image, intent: str, n_samples: int = 5, n_steps: int = 10,
                seed: int | None = None, max_new_tokens: int = 128) -> Prediction:
        """Predict the target camera pose for one image and intent.

        image is a file path or a PIL image. The defaults are the paper's: 5 pose
        samples, each integrated over 10 Euler steps. A seed fixes the sampling noise,
        and the generated text is deterministic either way. Poses aren't bit-exact across
        calls: the backbone's bf16 GPU kernels aren't deterministic on every GPU, which
        moved them by up to 0.007 on a GH200.
        """
        if not intent.strip():
            raise ValueError("intent must not be empty")
        if n_samples < 1 or n_steps < 1:
            raise ValueError("n_samples and n_steps must be at least 1")

        t0 = time.perf_counter()
        generation = self.backbone.generate(image, intent, max_new_tokens)
        t1 = time.perf_counter()

        hidden = generation.hidden.to(dtype=next(self.flow_head.parameters()).dtype)
        padding = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=hidden.device).manual_seed(seed)
        samples = np.stack([
            euler_integrate(self.flow_head, hidden, n_steps=n_steps,
                            kv_padding_mask=padding,
                            generator=generator).float().cpu().numpy().reshape(-1)
            for _ in range(n_samples)
        ])
        t2 = time.perf_counter()

        return Prediction(
            intent=intent,
            observation_gain=generation.text.strip(),
            pose9_mean=self.pose_norm.unnormalize(samples.mean(0)),
            pose9_std=samples.std(0).astype(np.float64) * self.pose_norm.std,
            pose9_samples=self.pose_norm.unnormalize(samples),
            timings={"generate_s": t1 - t0, "flow_head_s": t2 - t1},
        )
