# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nemotron_h import NemotronHAttentionDecoderLayer
from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHMTPAttentionDecoderLayer,
    _apply_first_step_block_replacement,
    get_mtp_inner_config,
)
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


def test_mtp_inner_config_applies_bottleneck_and_scaled_ffn_widths():
    config = NemotronHConfig(
        hidden_size=2688,
        intermediate_size=1856,
        moe_intermediate_size=1856,
        num_hidden_layers=4,
        hybrid_override_pattern="M-*E",
        mtp_hybrid_override_pattern="W-*E",
        num_nextn_predict_layers=1,
        mtp_bottleneck_hidden_size=1536,
        mtp_window_size=[1024, 0],
    )

    inner_config = get_mtp_inner_config(config)

    assert inner_config.hidden_size == 1536
    assert inner_config.intermediate_size == 1088
    assert inner_config.moe_intermediate_size == 1088
    assert inner_config.hybrid_override_pattern == "W-*E"
    assert inner_config.num_hidden_layers == 4
    assert config.hidden_size == 2688


def test_mtp_sliding_attention_requires_window_size():
    with pytest.raises(AssertionError, match="mtp_window_size"):
        NemotronHConfig(
            num_hidden_layers=1,
            hybrid_override_pattern="*",
            mtp_hybrid_override_pattern="W",
            num_nextn_predict_layers=1,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_mtp_decoder_layer_uses_mtp_rope_flag(monkeypatch, enabled):
    def parent_init(self, **kwargs):
        self.parent_kwargs = kwargs

    monkeypatch.setattr(NemotronHAttentionDecoderLayer, "__init__", parent_init)
    config = SimpleNamespace(mtp_use_rope=enabled)
    layer = NemotronHMTPAttentionDecoderLayer(config=config, layer_idx=0)

    assert layer.parent_kwargs["use_rope"] is enabled


def test_block_replacement_uses_learned_vector_for_each_offset():
    combined_input = torch.arange(32, dtype=torch.float32).view(8, 4)
    block_offsets = torch.tensor([-1, -1, 0, 1, -1, 0, 1, -1])
    replace_vectors = torch.tensor(
        [
            [1.0, 2.0, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
        ]
    )

    actual = _apply_first_step_block_replacement(
        combined_input=combined_input,
        parallel_drafting_block_offsets=block_offsets,
        first_step_block_replace_vectors=replace_vectors,
    )

    expected = combined_input.clone()
    for position, offset in ((2, 0), (3, 1), (5, 0), (6, 1)):
        expected[position] = replace_vectors[offset]

    torch.testing.assert_close(actual, expected)
