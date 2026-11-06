"""Guards which tensor the flow head sees, on a tiny random model, on any transformers."""
import inspect

import pytest
import torch
from conftest import PROMPT_LENGTH


def assert_matches_a_full_forward(backbone, generation):
    full, _ = backbone.hidden_states(generation.inputs, generation.full_ids)
    assert generation.hidden.shape == full.shape
    assert torch.allclose(generation.hidden, full, atol=1e-4)


@pytest.mark.parametrize("max_new_tokens", [1, 2, 8])
def test_generation_keeps_the_hidden_states_of_a_full_forward(backbone, max_new_tokens):
    generation = backbone.generate(None, "x", max_new_tokens)
    assert generation.full_ids.shape[1] == PROMPT_LENGTH + max_new_tokens
    # With an image, positions are not just the token index, so a wrong position for
    # the final token shows up here.
    assert backbone.model.model.rope_deltas.abs().sum() > 0
    assert_matches_a_full_forward(backbone, generation)


def test_stopping_on_eos_also_keeps_them(backbone):
    tokens = backbone.generate(None, "x", 8).full_ids[0, PROMPT_LENGTH:].tolist()
    config = backbone.model.generation_config
    previous = config.eos_token_id
    config.eos_token_id = tokens[2]
    try:
        generation = backbone.generate(None, "x", 8)
    finally:
        config.eos_token_id = previous
    assert generation.full_ids[0, -1] == tokens[2]
    assert generation.full_ids.shape[1] == PROMPT_LENGTH + tokens.index(tokens[2]) + 1
    assert_matches_a_full_forward(backbone, generation)


def test_hidden_states_are_taken_before_the_final_norm(backbone):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    hidden, mask = backbone.hidden_states({}, ids)
    assert hidden.shape == (1, 6, 32)
    assert torch.equal(mask, torch.ones_like(ids))

    text_model = backbone.model.model.language_model
    with torch.inference_mode():
        normed = backbone.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(text_model.norm(hidden), normed, atol=1e-5)
    assert not torch.allclose(hidden, normed, atol=1e-3)


def test_the_capture_hook_is_removed_afterwards(backbone):
    layer = backbone.model.model.language_model.layers[-1]
    before = len(layer._forward_hooks)
    backbone.hidden_states({}, torch.tensor([[1, 2, 3]]))
    backbone.generate(None, "x", 2)
    assert len(layer._forward_hooks) == before


def test_prompt_type_ids_are_padded_over_the_generated_tail(backbone):
    # Only transformers 5.x uses these; older versions must simply accept their absence.
    if "mm_token_type_ids" not in inspect.signature(backbone.model.forward).parameters:
        pytest.skip("this transformers version has no mm_token_type_ids")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    hidden, _ = backbone.hidden_states(
        {"mm_token_type_ids": torch.zeros((1, 3), dtype=torch.long)}, ids)
    assert hidden.shape == (1, 6, 32)
