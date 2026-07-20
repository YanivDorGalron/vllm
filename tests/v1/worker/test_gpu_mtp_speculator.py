# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.mtp import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


class _NoOpKernel:
    def __init__(self):
        self.kwargs = None

    def __getitem__(self, grid):
        def call(**kwargs):
            self.kwargs = kwargs

        return call


def test_parallel_expansion_preserves_prefix_and_bonus_token_shape(monkeypatch):
    speculator = object.__new__(MTPSpeculator)
    speculator.max_num_tokens = 10
    speculator.max_model_len = 32
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.num_extra_slots = 2
    speculator.parallel_drafting_token_id = 0
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(10, dtype=torch.int32),
        positions=torch.zeros(10, dtype=torch.int64),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        seq_lens=torch.zeros(2, dtype=torch.int32),
    )
    speculator.is_rejected = torch.zeros(10, dtype=torch.bool)
    speculator.is_masked = torch.zeros(10, dtype=torch.bool)
    speculator.sample_indices = torch.tensor([2, 3, 4, 6, 7, 8], dtype=torch.int32)
    speculator.hidden_mapping = torch.tensor([0, 1, 2, 5, 6, 7], dtype=torch.int32)
    speculator.block_offsets = torch.full((10,), -1, dtype=torch.int32)
    speculator.mask_block_offsets = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    speculator.hidden_states = torch.zeros((10, 2))
    speculator.req_arange = torch.arange(3, dtype=torch.int32)

    expand_kernel = _NoOpKernel()
    monkeypatch.setattr(
        spec_module,
        "copy_and_expand_eagle_inputs_kernel",
        expand_kernel,
    )
    monkeypatch.setattr(
        spec_module.triton,
        "next_power_of_2",
        lambda value: 1 << (value - 1).bit_length(),
        raising=False,
    )

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=6,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        input_ids=torch.zeros(6, dtype=torch.int32),
        positions=torch.arange(6, dtype=torch.int64),
    )

    speculator._expand_parallel_inputs(
        input_batch=input_batch,
        target_hidden_states=torch.zeros((6, 2)),
        num_sampled=torch.tensor([1, 0], dtype=torch.int32),
        num_rejected=torch.tensor([0, 1], dtype=torch.int32),
        # Request-state tokens are stored as [max_num_reqs, 1]. Do not let
        # torch.where broadcast this into a [num_reqs, num_reqs] matrix.
        last_sampled=torch.tensor([[101], [202]], dtype=torch.int64),
        next_prefill_tokens=torch.tensor([301, 302], dtype=torch.int64),
    )

    assert expand_kernel.kwargs is not None
    assert (
        expand_kernel.kwargs["out_parallel_drafting_block_offsets_ptr"]
        is speculator.block_offsets
    )
    bonus_tokens = expand_kernel.kwargs["next_token_ids_ptr"]
    assert bonus_tokens.shape == (2,)
    assert torch.equal(bonus_tokens, torch.tensor([101, 302], dtype=torch.int32))

    assert torch.equal(
        speculator.input_buffers.seq_lens,
        torch.tensor([12, 22], dtype=torch.int32),
    )
    original_query_lens = torch.diff(input_batch.query_start_loc)
    expanded_query_lens = torch.diff(speculator.input_buffers.query_start_loc)
    assert torch.equal(
        speculator.input_buffers.seq_lens - expanded_query_lens,
        input_batch.seq_lens - original_query_lens,
    )


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
