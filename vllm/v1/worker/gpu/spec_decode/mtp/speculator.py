# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

logger = init_logger(__name__)


class MTPSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.parallel_drafting = self.speculative_config.parallel_drafting
        if not self.parallel_drafting:
            return

        assert self.speculative_config.use_nemotron_h_mtp()
        hf_config = self.draft_model_config.hf_config
        assert getattr(hf_config, "mtp_naive_parallel_enabled", False)

        self.num_extra_slots = self.num_speculative_steps - 1
        parallel_token_id = getattr(hf_config, "pad_token_id", 0)
        self.parallel_drafting_token_id = (
            parallel_token_id
            if isinstance(parallel_token_id, int) and parallel_token_id >= 0
            else 0
        )

        self.is_rejected = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=device
        )
        self.is_masked = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=device
        )
        self.block_offsets = torch.full(
            (self.max_num_tokens,), -1, dtype=torch.int32, device=device
        )
        self.hidden_mapping = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )

        max_num_samples = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_samples, dtype=torch.int32, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_samples, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)
        self.mask_block_offsets = torch.arange(
            self.num_extra_slots, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)
        self.req_arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if not self.parallel_drafting:
            return super().init_cudagraph_manager(cudagraph_mode)
        if cudagraph_mode != CUDAGraphMode.NONE:
            logger.warning(
                "Nemotron-H parallel MTP uses one eager draft forward; "
                "target-model CUDA graphs remain enabled."
            )

    def capture(self, attn_states: dict | None = None) -> None:
        if not self.parallel_drafting:
            assert attn_states is not None
            return super().capture(attn_states)

    def _get_additional_model_kwargs(self, num_tokens: int) -> dict[str, Any]:
        if not self.parallel_drafting:
            return {}
        return {"parallel_drafting_block_offsets": self.block_offsets[:num_tokens]}

    def _expand_parallel_inputs(
        self,
        input_batch: InputBatch,
        target_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_output_tokens = num_target_tokens + num_reqs * self.num_extra_slots
        if num_output_tokens > self.max_num_tokens:
            raise ValueError(
                "Nemotron-H parallel draft batch exceeds max_num_batched_tokens: "
                f"{num_output_tokens} > {self.max_num_tokens}."
            )

        req_state = input_batch.idx_mapping[:num_reqs].to(torch.long)
        # last_sampled is stored as [max_num_reqs, 1]. Flatten both token
        # sources so torch.where cannot broadcast across request rows.
        sampled_tokens = last_sampled.reshape(-1)[req_state]
        prefill_tokens = next_prefill_tokens.reshape(-1)[req_state]
        bonus_tokens = torch.where(
            num_sampled[:num_reqs] > 0,
            sampled_tokens,
            prefill_tokens,
        ).to(torch.int32)
        valid_query_end = input_batch.query_start_loc[1 : num_reqs + 1] - 1
        valid_query_end = valid_query_end - num_rejected[:num_reqs]

        max_tokens_per_req = (
            int(input_batch.num_scheduled_tokens.max()) + self.num_extra_slots
        )
        block_size = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
        num_blocks = triton.cdiv(max_tokens_per_req, block_size)
        copy_and_expand_eagle_inputs_kernel[(num_reqs, num_blocks)](
            target_token_ids_ptr=input_batch.input_ids,
            target_positions_ptr=input_batch.positions,
            next_token_ids_ptr=bonus_tokens,
            out_input_ids_ptr=self.input_buffers.input_ids,
            out_positions_ptr=self.input_buffers.positions,
            out_is_rejected_token_mask_ptr=self.is_rejected,
            out_is_masked_token_mask_ptr=self.is_masked,
            out_parallel_drafting_block_offsets_ptr=self.block_offsets,
            out_new_token_indices_ptr=self.sample_indices,
            out_hidden_state_mapping_ptr=self.hidden_mapping,
            query_start_loc_ptr=input_batch.query_start_loc,
            query_end_loc_ptr=valid_query_end,
            padding_token_id=0,
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            total_input_tokens=num_target_tokens,
            num_padding_slots_per_request=self.num_speculative_steps,
            shift_input_ids=True,
            BLOCK_SIZE_TOKENS=block_size,
        )

        num_samples = num_reqs * self.num_speculative_steps
        self.block_offsets[:num_output_tokens].fill_(-1)
        if self.num_extra_slots:
            masked_indices = self.sample_indices[:num_samples].view(
                num_reqs, self.num_speculative_steps
            )[:, 1:]
            self.block_offsets[masked_indices.reshape(-1).to(torch.long)] = (
                self.mask_block_offsets[: num_reqs * self.num_extra_slots]
            )

        self.hidden_states[:num_output_tokens].zero_()
        hidden_destinations = self.hidden_mapping[:num_target_tokens].to(torch.long)
        self.hidden_states[hidden_destinations] = target_hidden_states[
            :num_target_tokens
        ]
        clear_hidden = torch.logical_or(
            self.is_masked[:num_output_tokens],
            self.is_rejected[:num_output_tokens],
        )
        self.hidden_states[:num_output_tokens].masked_fill_(
            clear_hidden.unsqueeze(-1), 0
        )

        expanded_query_start = (
            input_batch.query_start_loc[: num_reqs + 1]
            + self.num_extra_slots * self.req_arange[: num_reqs + 1]
        )
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(expanded_query_start)
        self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(num_output_tokens)
        self.input_buffers.seq_lens[:num_reqs].copy_(
            (input_batch.seq_lens[:num_reqs] + self.num_extra_slots).clamp(
                max=self.max_model_len
            )
        )
        self.input_buffers.seq_lens[num_reqs:].zero_()
        return num_output_tokens, expanded_query_start

    def _prepare_parallel_attention(
        self,
        input_batch: InputBatch,
        num_output_tokens: int,
        expanded_query_start: torch.Tensor,
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping[:num_reqs]

        # Gather in active-batch order before building attention metadata. This is
        # essential when requests are reordered or concurrency is greater than one.
        self.block_tables.gather_block_tables(idx_mapping, num_reqs)

        positions = self.input_buffers.positions[:num_output_tokens]
        exceeds_max_len = positions >= self.max_model_len
        positions.clamp_(max=self.max_model_len - 1)
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping,
            expanded_query_start,
            positions,
            num_output_tokens,
        )
        invalid_slots = torch.logical_or(
            self.is_rejected[:num_output_tokens], exceeds_max_len
        )
        slot_mappings.masked_fill_(invalid_slots.unsqueeze(0), PAD_SLOT_ID)

        expanded_lens_np = (
            input_batch.num_scheduled_tokens[:num_reqs] + self.num_extra_slots
        )
        query_start_cpu = torch.from_numpy(
            np.concatenate(([0], np.cumsum(expanded_lens_np, dtype=np.int32)))
        )
        max_query_len = int(expanded_lens_np.max())
        seq_lens_cpu_upper_bound = (
            input_batch.seq_lens_cpu_upper_bound[:num_reqs] + self.num_extra_slots
        ).clamp(max=self.max_model_len)
        self.draft_max_seq_len = int(seq_lens_cpu_upper_bound.max().item())

        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_output_tokens,
            query_start_loc_gpu=expanded_query_start,
            query_start_loc_cpu=query_start_cpu,
            max_query_len=max_query_len,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=self.draft_max_seq_len,
            block_tables=[
                table[:num_reqs] for table in self.block_tables.input_block_tables
            ],
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            positions=positions,
            causal=True,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        return attn_metadata, slot_mappings_by_layer

    def _dummy_parallel_propose(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        num_samples = num_reqs * self.num_speculative_steps

        self.hidden_states[:num_tokens].copy_(last_hidden_states[:num_tokens])
        self.input_buffers.input_ids[:num_tokens].copy_(
            input_batch.input_ids[:num_tokens]
        )
        self.input_buffers.positions[:num_tokens].copy_(
            input_batch.positions[:num_tokens]
        )
        self.block_offsets[:num_tokens].fill_(-1)

        self._prepare_eplb_forward(num_tokens)
        output, _ = self._run_model(
            num_tokens,
            attn_metadata=None,
            slot_mappings=None,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        sample_hidden = output[:num_samples]
        sample_positions = self.input_buffers.positions[:num_samples]
        sample_idx_mapping = self.idx_mapping[:num_reqs].repeat_interleave(
            self.num_speculative_steps
        )
        sampled = self.sample_draft(
            sample_hidden,
            sample_positions,
            sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.sample_col[:num_samples],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs].copy_(
            sampled.view(num_reqs, self.num_speculative_steps)
        )
        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if not self.parallel_drafting:
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )
        if aux_hidden_states:
            raise NotImplementedError(
                "Nemotron-H parallel MTP does not consume Eagle3 aux states."
            )
        if mm_inputs is not None:
            raise NotImplementedError(
                "Nemotron-H parallel MTP currently supports text-only inputs."
            )

        num_reqs = input_batch.num_reqs
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)

        if dummy_run and (skip_attn_for_dummy_run or is_profile):
            return self._dummy_parallel_propose(
                input_batch,
                last_hidden_states,
                num_tokens_across_dp,
            )

        num_tokens, expanded_query_start = self._expand_parallel_inputs(
            input_batch,
            last_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )
        draft_attn_metadata, draft_slots = self._prepare_parallel_attention(
            input_batch, num_tokens, expanded_query_start
        )

        _, parallel_tokens_across_dp = dispatch_cg_and_sync_dp(
            cudagraph_manager=None,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_token_count=None,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=True,
        )
        self._prepare_eplb_forward(num_tokens)
        output, _ = self._run_model(
            num_tokens,
            draft_attn_metadata,
            draft_slots,
            num_tokens_across_dp=parallel_tokens_across_dp,
        )

        num_samples = num_reqs * self.num_speculative_steps
        indices = self.sample_indices[:num_samples].to(torch.long)
        sample_hidden = output[indices]
        sample_positions = self.input_buffers.positions[indices]
        self.sample_idx_mapping[:num_samples].copy_(
            self.idx_mapping[:num_reqs].repeat_interleave(self.num_speculative_steps)
        )
        sampled = self.sample_draft(
            sample_hidden,
            sample_positions,
            self.sample_idx_mapping[:num_samples],
            self.temperature,
            self.seeds,
            self.sample_col[:num_samples],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs].copy_(
            sampled.view(num_reqs, self.num_speculative_steps)
        )
        return self.draft_tokens[:num_reqs]
