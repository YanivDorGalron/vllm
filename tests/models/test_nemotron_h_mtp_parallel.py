# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHAttentionDecoderLayer,
    NemotronHMTPAttentionDecoderLayer,
    _apply_first_step_block_replacement,
)

pytestmark = pytest.mark.cpu_test


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
