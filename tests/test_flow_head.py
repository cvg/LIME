import pytest
import torch
from torch import nn

from lime_camera.models.flow_head import POSE_DIM, FlowMatchingPoseHead, euler_integrate


class ConstantHead(nn.Module):
    """A head that always outputs the same thing, recording the x it was given."""

    def __init__(self, output: torch.Tensor, predict_x: bool):
        super().__init__()
        self.output, self.predict_x, self.seen = output, predict_x, []

    def forward(self, vlm_hidden, x_t, t, kv_padding_mask=None):
        self.seen.append(x_t.clone())
        return self.output


def small_head(**kwargs) -> FlowMatchingPoseHead:
    return FlowMatchingPoseHead(vlm_hidden_size=16, embed_dim=32, n_layers=2, n_heads=2,
                                time_embed_dim=16, **kwargs).eval()


def test_parameter_names_match_checkpoints():
    # Checkpoints load strictly, so renaming a module here would break them.
    head = FlowMatchingPoseHead(vlm_hidden_size=16, embed_dim=32, n_layers=1, n_heads=2,
                                time_embed_dim=16)
    assert set(head.state_dict()) == {
        "action_in.weight", "action_in.bias",
        "time_in.1.weight", "time_in.1.bias", "time_in.3.weight", "time_in.3.bias",
        "vlm_proj.weight", "vlm_proj.bias",
        "blocks.0.norm_q.weight", "blocks.0.norm_q.bias",
        "blocks.0.norm_kv.weight", "blocks.0.norm_kv.bias",
        "blocks.0.attn.in_proj_weight", "blocks.0.attn.in_proj_bias",
        "blocks.0.attn.out_proj.weight", "blocks.0.attn.out_proj.bias",
        "blocks.0.norm_ffn.weight", "blocks.0.norm_ffn.bias",
        "blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias",
        "blocks.0.ffn.2.weight", "blocks.0.ffn.2.bias",
        "norm_out.weight", "norm_out.bias",
        "action_out.weight", "action_out.bias",
    }


def test_paper_architecture_has_the_expected_size():
    head = FlowMatchingPoseHead(vlm_hidden_size=2560, embed_dim=512, n_layers=6, n_heads=8,
                                time_embed_dim=256)
    assert sum(p.numel() for p in head.parameters()) == 20_636_681


def test_output_starts_at_zero():
    out = small_head()(torch.randn(2, 5, 16), torch.randn(2, POSE_DIM), torch.rand(2))
    assert out.shape == (2, POSE_DIM)
    assert torch.equal(out, torch.zeros_like(out))


def test_velocity_mode_integrates_the_output():
    velocity = torch.full((1, POSE_DIM), 2.0)
    head = ConstantHead(velocity, predict_x=False)
    generator = torch.Generator().manual_seed(0)
    start = torch.randn(1, POSE_DIM, generator=torch.Generator().manual_seed(0))
    x = euler_integrate(head, torch.zeros(1, 3, 4), n_steps=10, generator=generator)
    assert torch.allclose(x, start + velocity)


def test_x_prediction_walks_a_straight_line_to_the_target():
    target = torch.full((1, POSE_DIM), 3.0)
    head = ConstantHead(target, predict_x=True)
    start = torch.randn(1, POSE_DIM, generator=torch.Generator().manual_seed(0))
    out = euler_integrate(head, torch.zeros(1, 3, 4), n_steps=10,
                          generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, target)
    for k, x_t in enumerate(head.seen):
        t = k / 10
        assert torch.allclose(x_t, (1 - t) * start + t * target, atol=1e-5)


def test_padded_tokens_do_not_affect_the_output():
    head = small_head()
    nn.init.normal_(head.action_out.weight)     # undo the zero init
    hidden = torch.randn(1, 6, 16)
    changed = hidden.clone()
    changed[:, 4:] += 5.0
    mask = torch.tensor([[False, False, False, False, True, True]])
    x, t = torch.randn(1, POSE_DIM), torch.tensor([0.3])
    with torch.no_grad():
        assert torch.allclose(head(hidden, x, t, kv_padding_mask=mask),
                              head(changed, x, t, kv_padding_mask=mask))
        assert not torch.allclose(head(hidden, x, t), head(changed, x, t))


@pytest.mark.parametrize("predict_x", [False, True])
def test_generator_makes_sampling_reproducible(predict_x):
    head = small_head(predict_x=predict_x)
    nn.init.normal_(head.action_out.weight)
    hidden = torch.randn(1, 4, 16)

    def run(seed):
        return euler_integrate(head, hidden, n_steps=3,
                               generator=torch.Generator().manual_seed(seed))

    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1))
