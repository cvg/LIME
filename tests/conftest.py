"""A tiny random Qwen3-VL and a stand-in processor: no weights or GPU needed."""
import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lime_camera.models.backbone import Backbone

IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END = 60, 61, 62, 63
PROMPT_LENGTH = 10


class TinyProcessor:
    """Stands in for the real one: a prompt with a 4-token image in the middle."""
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)

    def __init__(self, model):
        self.needs_type_ids = "mm_token_type_ids" in inspect.signature(model.forward).parameters

    def apply_chat_template(self, *_args, **_kwargs):
        ids = torch.tensor([[1, 2, VISION_START, *[IMAGE_TOKEN] * 4, VISION_END, 3, 4]])
        pixels = torch.randn(16, 3 * 2 * 16 * 16, generator=torch.Generator().manual_seed(0))
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 4, 4]])}
        if self.needs_type_ids:
            inputs["mm_token_type_ids"] = (ids == IMAGE_TOKEN).long()
        return inputs

    def batch_decode(self, ids, **_kwargs):
        return [" ".join(str(i) for i in ids[0].tolist())]


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "max_position_embeddings": 128, "head_dim": 16,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4]},
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "out_hidden_size": 32, "deepstack_visual_indexes": [],
        },
        image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    return Qwen3VLForConditionalGeneration(config).eval()


@pytest.fixture(scope="module")
def backbone():
    model = tiny_model()
    return Backbone(model, TinyProcessor(model))
