# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMLPDecoderLayer,
    NemotronHMoEDecoderLayer,
)
from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHMTPAttentionDecoderLayer,
    NemotronHMTPMLPDecoderLayer,
    NemotronHMTPMoEDecoderLayer,
    _get_mtp_inner_config,
    _get_mtp_layer_config,
)
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


def make_config(
    *,
    pattern: str = "W-*E",
    bottleneck_hidden_size: int | None = 1920,
    window_size: tuple[int, int] | list[int] | None = (1024, 0),
) -> NemotronHConfig:
    return NemotronHConfig(
        hidden_size=2688,
        intermediate_size=1856,
        num_hidden_layers=2,
        hybrid_override_pattern="M-",
        mtp_hybrid_override_pattern=pattern,
        mtp_bottleneck_hidden_size=bottleneck_hidden_size,
        mtp_window_size=window_size,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=2,
        n_routed_experts=128,
        n_shared_experts=1,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
    )


@pytest.mark.parametrize(
    ("hidden_size", "expected_ffn"),
    [
        (1280, 896),
        (1344, 928),
        (1408, 992),
        (1536, 1088),
        (1664, 1152),
        (1920, 1344),
        (2560, 1792),
    ],
)
def test_mtp_inner_config_scales_dense_and_moe_widths(
    hidden_size: int, expected_ffn: int
) -> None:
    config = make_config(bottleneck_hidden_size=hidden_size)

    inner_config = _get_mtp_inner_config(config)

    assert inner_config.hidden_size == hidden_size
    assert inner_config.intermediate_size == expected_ffn
    assert inner_config.moe_intermediate_size == expected_ffn
    assert inner_config.moe_shared_expert_intermediate_size == 3712
    assert inner_config.hybrid_override_pattern == "*-*E"
    assert inner_config.num_hidden_layers == 4


def test_mtp_inner_config_preserves_legacy_width_without_bottleneck() -> None:
    config = make_config(
        pattern="*E",
        bottleneck_hidden_size=None,
        window_size=None,
    )

    inner_config = _get_mtp_inner_config(config)

    assert inner_config.hidden_size == config.hidden_size
    assert inner_config.intermediate_size == config.intermediate_size
    assert inner_config.moe_intermediate_size == config.moe_intermediate_size
    assert inner_config.hybrid_override_pattern == "*E"


def test_mtp_layer_config_applies_window_only_to_w_layers() -> None:
    config = make_config(window_size=[2048, 0])
    inner_config = _get_mtp_inner_config(config)

    local_config = _get_mtp_layer_config(inner_config, config, "W")
    global_config = _get_mtp_layer_config(inner_config, config, "*")
    dense_config = _get_mtp_layer_config(inner_config, config, "-")

    assert local_config.sliding_window == 2049
    assert global_config.sliding_window is None
    assert dense_config.sliding_window is None


@pytest.mark.parametrize(
    "pattern",
    [
        "*E",
        "W-*-",
        "W-W-*-",
        "W-W-W-*-",
        "WWW*E",
        "W-W-*E",
        "W-W-W-*E",
        "WWW*",
        "WE*E",
        "WEWE*E",
        "WEWEWE*E",
    ],
)
def test_mtp_experiment_patterns_are_supported(pattern: str) -> None:
    window_size = [1024, 0] if "W" in pattern else None

    config = make_config(pattern=pattern, window_size=window_size)
    inner_config = _get_mtp_inner_config(config)

    assert inner_config.hybrid_override_pattern == pattern.replace("W", "*")
    assert inner_config.num_hidden_layers == len(pattern)


@pytest.mark.parametrize(
    ("pattern", "window_size"),
    [
        ("W-*E", None),
        ("W-*E", [1024]),
        ("W-*E", [0, 0]),
        ("W-*E", [1024, 1]),
    ],
)
def test_mtp_window_validation(pattern: str, window_size: list[int] | None) -> None:
    with pytest.raises(ValueError, match="mtp_window_size"):
        make_config(pattern=pattern, window_size=window_size)


def test_mtp_pattern_validation() -> None:
    with pytest.raises(ValueError, match="mtp_hybrid_override_pattern"):
        make_config(pattern="WM*E")


def _make_uninitialized_mtp_layer(layer_cls: type[torch.nn.Module]) -> torch.nn.Module:
    layer = layer_cls.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    layer.has_start_projections = False
    layer.has_end_norm = False
    return layer


def test_mtp_attention_layer_delegates_to_attention_forward(monkeypatch) -> None:
    calls = []

    def attention_forward(self, positions, hidden_states, residual, **kwargs):
        calls.append((positions, hidden_states, residual))
        return hidden_states + 1, residual

    monkeypatch.setattr(
        NemotronHAttentionDecoderLayer,
        "forward",
        attention_forward,
    )
    layer = _make_uninitialized_mtp_layer(NemotronHMTPAttentionDecoderLayer)
    positions = torch.tensor([0])
    hidden_states = torch.zeros(1, 2)

    output, residual = layer(
        inputs_embeds=hidden_states,
        positions=positions,
        hidden_states=hidden_states,
    )

    assert calls == [(positions, hidden_states, None)]
    assert torch.equal(output, hidden_states + 1)
    assert residual is None


@pytest.mark.parametrize(
    ("mtp_layer_cls", "decoder_layer_cls"),
    [
        (NemotronHMTPMLPDecoderLayer, NemotronHMLPDecoderLayer),
        (NemotronHMTPMoEDecoderLayer, NemotronHMoEDecoderLayer),
    ],
)
def test_mtp_ffn_layers_delegate_to_their_decoder_forward(
    monkeypatch,
    mtp_layer_cls,
    decoder_layer_cls,
) -> None:
    calls = []

    def decoder_forward(self, hidden_states, residual, **kwargs):
        calls.append((hidden_states, residual))
        return hidden_states + 1, residual

    monkeypatch.setattr(decoder_layer_cls, "forward", decoder_forward)
    layer = _make_uninitialized_mtp_layer(mtp_layer_cls)
    hidden_states = torch.zeros(1, 2)

    output, residual = layer(
        inputs_embeds=hidden_states,
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
    )

    assert calls == [(hidden_states, None)]
    assert torch.equal(output, hidden_states + 1)
    assert residual is None
