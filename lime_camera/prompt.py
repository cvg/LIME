"""The chat turn the model sees."""
from __future__ import annotations

from pathlib import Path


def build_messages(image, intent: str) -> list[dict]:
    """Return the chat messages for one image and intent.

    The user turn is the image plus the bare intent text, with no preamble: that is
    the prompt the model expects. The model then writes the assistant turn, the
    observation-gain text.
    """
    return [{"role": "user", "content": [
        {"type": "image", "image": str(image) if isinstance(image, Path) else image},
        {"type": "text", "text": intent},
    ]}]
