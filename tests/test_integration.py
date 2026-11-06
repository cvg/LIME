"""End-to-end checks on real weights. Set LIME_CHECKPOINT to run them; needs a GPU.

Exact generated text is not compared across environments: greedy decoding flips on
tiny numeric differences, even between attention backends on one machine. Repeat runs
in one environment are identical, and poses are compared whenever the text matches.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = os.environ.get("LIME_CHECKPOINT")
IMAGE = ROOT / "examples" / "living_room.jpg"     # 1920x1440
GOLDEN = json.loads((Path(__file__).parent / "golden_poses.json").read_text())

pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="LIME_CHECKPOINT is not set")


@pytest.fixture(scope="module")
def model():
    from lime_camera import LIME
    return LIME.from_pretrained(CHECKPOINT)


@pytest.mark.parametrize("golden", GOLDEN, ids=[g["intent"] for g in GOLDEN])
def test_poses_match_when_the_text_matches(model, golden):
    p = model.predict(IMAGE, golden["intent"], seed=0)
    assert p.pose9_samples.shape == (5, 9)
    if p.observation_gain == golden["observation_gain"]:
        # Measured across transformers 4.57.0 to 5.22.1: at most 0.014 m.
        moved = np.linalg.norm(p.pose9_mean[:3] - np.array(golden["pose9_mean"][:3]))
        assert moved < 0.1, f"pose moved {moved:.3f} m with identical text"


# The backbone's bf16 kernels aren't deterministic on every GPU: on a GH200, repeated calls
# with one seed moved poses by up to 0.007. Different seeds move them by centimetres.
SAME_SEED_ATOL = 0.02


def test_seed_controls_the_samples_and_text_is_deterministic(model):
    a = model.predict(IMAGE, "Find my bike.", seed=1)
    b = model.predict(IMAGE, "Find my bike.", seed=1)
    c = model.predict(IMAGE, "Find my bike.", seed=2)
    assert a.observation_gain == b.observation_gain == c.observation_gain
    np.testing.assert_allclose(a.pose9_samples, b.pose9_samples, rtol=0, atol=SAME_SEED_ATOL)
    assert np.abs(a.pose9_samples - c.pose9_samples).max() > SAME_SEED_ATOL


def test_k_and_steps_are_respected(model):
    assert model.predict(IMAGE, "Find my bike.", n_samples=3,
                         n_steps=2, seed=0).pose9_samples.shape == (3, 9)


def test_outputs_are_valid_poses(model):
    p = model.predict(IMAGE, "Find my bike.", seed=0)
    for T in [p.T_mean, *p.T_samples]:
        assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
        assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(p.T_mean[:3, 3], p.pose9_mean[:3])


def test_path_and_pil_inputs_agree(model):
    a = model.predict(IMAGE, "Find my bike.", seed=0)
    b = model.predict(Image.open(IMAGE), "Find my bike.", seed=0)
    assert a.observation_gain == b.observation_gain
    np.testing.assert_allclose(a.pose9_samples, b.pose9_samples, rtol=0, atol=SAME_SEED_ATOL)


def test_bad_arguments_are_rejected(model):
    with pytest.raises(ValueError):
        model.predict(IMAGE, "   ")
    with pytest.raises(ValueError):
        model.predict(IMAGE, "x", n_samples=0)


def test_image_budget_comes_from_the_checkpoint(model):
    inputs = model.backbone.generate(IMAGE, "x", max_new_tokens=1).inputs
    # 200704 max pixels with 16 px patches, merged 2x2: a 1920x1440 image is resized to
    # 512x384, which gives 192 tokens.
    assert inputs["image_grid_thw"][0].tolist() == [1, 24, 32]


def test_head_sees_the_pre_norm_tensor(model):
    hidden = model.backbone.generate(IMAGE, "Find my bike.").hidden
    # Pre-norm activations run around 5 on this checkpoint; post-norm around 2.
    assert 4.0 < hidden.float().abs().mean().item() < 7.0


def test_cached_hidden_states_match_a_full_forward(model):
    backbone = model.backbone
    generation = backbone.generate(IMAGE, "Find my bike.")
    full, _ = backbone.hidden_states(generation.inputs, generation.full_ids)
    cached, full = generation.hidden.float(), full.float()
    # In bf16, rounding differs between the two paths: measured up to 5% overall and
    # 8% on the final token. In float32, they agree to about 3e-6.
    assert (cached - full).norm() / full.norm() < 0.1
    assert (cached[:, -1] - full[:, -1]).norm() / full[:, -1].norm() < 0.15


def test_cli_writes_json_to_stdout():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "predict.py"), "--checkpoint", CHECKPOINT,
         "--image", str(IMAGE), "--intent", "Find my bike.", "--seed", "0",
         "--samples", "2"],
        capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert output["intent"] == "Find my bike."
    assert np.array(output["T_samples"]).shape == (2, 4, 4)
