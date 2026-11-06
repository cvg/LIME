"""Thin wrapper around the Qwen3-VL backbone.

The model code itself lives in `transformers`. This file only holds what LIME
needs from it: generating the observation-gain text, and reading the hidden
states the flow head conditions on.

The head conditions on the last decoder layer's output, BEFORE the text model's
final norm. `output_hidden_states[-1]` returned exactly that when the head was
trained (transformers 4.57), but 5.x returns the normalized tensor there, which
runs without error and shifts every pose. A hook on the layer means the same thing
on every version.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..prompt import build_messages

logger = logging.getLogger(__name__)


@contextmanager
def last_layer_outputs(model):
    """Collect the last decoder layer's output from every forward inside the block."""
    outputs = []

    def capture(_module, _inputs, output):
        outputs.append(output[0] if isinstance(output, tuple) else output)

    handle = model.model.language_model.layers[-1].register_forward_hook(capture)
    try:
        yield outputs
    finally:
        handle.remove()


@dataclass
class Generation:
    text: str                 # the observation-gain text
    hidden: torch.Tensor      # (1, L, H) what the flow head conditions on
    inputs: dict              # the processor's output for the prompt
    full_ids: torch.Tensor    # (1, L) the prompt followed by the generated tokens


class Backbone:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, checkpoint, dtype=torch.bfloat16,
                        attn_implementation: str = "sdpa") -> "Backbone":
        """Load a fine-tuned backbone and its processor.

        The image budget comes from the checkpoint's preprocessor config. It must
        match what the checkpoint was trained at, because a different budget gives
        the vision tower a different token count and degrades predictions without
        any error.
        """
        logger.info("loading backbone from %s", checkpoint)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint, dtype=dtype, attn_implementation=attn_implementation,
            device_map="auto",
        )
        model.eval()
        return cls(model, AutoProcessor.from_pretrained(checkpoint))

    @property
    def device(self):
        return self.model.device

    def generate(self, image, intent: str, max_new_tokens: int = 128) -> Generation:
        """Generate the observation-gain text, keeping the hidden states it produced.

        Generation forwards the whole prompt once and then each new token against the
        KV cache, so the hidden states of every token are already there, except for
        the final one: generation stops as soon as it has that token. One more cached
        step fills in the gap.
        """
        inputs = self.processor.apply_chat_template(
            build_messages(image, intent), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        tokenizer = self.processor.tokenizer
        with torch.inference_mode(), last_layer_outputs(self.model) as outputs:
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            full_ids = generated.sequences
            # The position is explicit because the model's own default for a cached
            # step depends on the transformers version: 4.57 needs cache_position or it
            # uses 0, and 5.x builds one position per token of the attention mask.
            position = full_ids.shape[1] - 1 + self.model.model.rope_deltas
            self.model(
                input_ids=full_ids[:, -1:],
                attention_mask=torch.ones_like(full_ids),
                position_ids=position.to(self.device).view(1, 1, 1).expand(3, 1, 1),
                past_key_values=generated.past_key_values,
                use_cache=True,
            )
        # Under device_map="auto" the last layer can sit on a different GPU.
        hidden = torch.cat([o.to(self.device) for o in outputs], dim=1)
        if hidden.shape[1] != full_ids.shape[1]:
            raise RuntimeError(f"captured hidden states for {hidden.shape[1]} tokens, "
                               f"but the sequence has {full_ids.shape[1]}")
        text = self.processor.batch_decode(
            full_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return Generation(text, hidden, inputs, full_ids)

    def hidden_states(self, prompt_inputs: dict, full_ids: torch.Tensor):
        """Forward [prompt + text] in one full pass.

        Prediction doesn't call this, because `generate` already has these hidden
        states. The tests use it to check that the two agree.

        Returns (hidden (B, L, H), attention_mask (B, L)).
        """
        full_ids = full_ids.to(self.device)
        fwd = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids, device=self.device),
        }
        for k in ("pixel_values", "image_grid_thw",
                  "pixel_values_videos", "video_grid_thw"):
            if prompt_inputs.get(k) is not None:
                fwd[k] = prompt_inputs[k].to(self.device)
        # transformers 5.x requires mm_token_type_ids alongside image_grid_thw for
        # M-RoPE. The processor returns it only for the prompt; pad zeros (text
        # tokens) over the generated tail so the length matches full_ids.
        if prompt_inputs.get("mm_token_type_ids") is not None:
            mm = prompt_inputs["mm_token_type_ids"].to(self.device)
            pad = torch.zeros((mm.shape[0], full_ids.shape[1] - mm.shape[1]),
                              dtype=mm.dtype, device=self.device)
            fwd["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)

        with torch.inference_mode(), last_layer_outputs(self.model) as outputs:
            self.model(**fwd, use_cache=False)
        return outputs[0].to(self.device), fwd["attention_mask"]
