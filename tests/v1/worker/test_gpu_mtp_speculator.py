# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.mtp import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def test_parallel_attention_gathers_active_request_block_tables(monkeypatch):
    speculator = object.__new__(MTPSpeculator)
    speculator.device = torch.device("cpu")
    speculator.num_extra_slots = 1
    speculator.max_model_len = 32
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(5, dtype=torch.int64),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
    )
    speculator.is_rejected = torch.zeros(5, dtype=torch.bool)
    speculator.attn_groups = []
    speculator.kv_cache_config = MagicMock()

    slot_mappings = torch.zeros((1, 5), dtype=torch.int64)
    block_tables = MagicMock()
    block_tables.input_block_tables = [torch.zeros((2, 4), dtype=torch.int32)]
    block_tables.compute_slot_mappings.return_value = slot_mappings
    speculator.block_tables = block_tables

    monkeypatch.setattr(spec_module, "build_attn_metadata", lambda **kwargs: {})
    monkeypatch.setattr(
        spec_module,
        "build_slot_mappings_by_layer",
        lambda slots, config: {"layer": slots},
    )

    idx_mapping = torch.tensor([3, 1], dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=idx_mapping,
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([3, 2], dtype=torch.int32),
        prompt_lens=None,
    )
    expanded_query_start = torch.tensor([0, 3, 5], dtype=torch.int32)

    speculator._prepare_parallel_attention(
        input_batch,
        num_output_tokens=5,
        expanded_query_start=expanded_query_start,
    )

    gather_args = block_tables.gather_block_tables.call_args.args
    assert torch.equal(gather_args[0], idx_mapping)
    assert gather_args[1] == 2

    slot_args = block_tables.compute_slot_mappings.call_args.args
    assert torch.equal(slot_args[0], idx_mapping)
    assert torch.equal(slot_args[1], expanded_query_start)
    assert torch.equal(slot_args[2], speculator.input_buffers.positions)
    assert slot_args[3] == 5
