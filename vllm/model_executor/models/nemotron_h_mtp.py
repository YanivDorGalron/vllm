# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import copy
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

from .interfaces import SupportsPP
from .nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMLPDecoderLayer,
    NemotronHMoEDecoderLayer,
)


def _apply_first_step_block_replacement(
    combined_input: torch.Tensor,
    parallel_drafting_block_offsets: torch.Tensor | None,
    first_step_block_replace_vectors: torch.Tensor | None,
) -> torch.Tensor:
    if (
        first_step_block_replace_vectors is None
        or parallel_drafting_block_offsets is None
    ):
        return combined_input

    if parallel_drafting_block_offsets.ndim != 1:
        raise ValueError(
            "Expected 1D parallel_drafting_block_offsets for Nemotron-H MTP "
            f"parallel drafting, got shape "
            f"{tuple(parallel_drafting_block_offsets.shape)}."
        )
    if parallel_drafting_block_offsets.shape[0] != combined_input.shape[0]:
        raise ValueError(
            "parallel_drafting_block_offsets must align with the flattened token "
            f"dimension. Got {parallel_drafting_block_offsets.shape[0]=} and "
            f"{combined_input.shape[0]=}."
        )
    num_replace_vectors = first_step_block_replace_vectors.shape[0]
    if num_replace_vectors == 0:
        return combined_input

    valid_mask = parallel_drafting_block_offsets.unsqueeze(-1).ge(0)
    safe_offsets = parallel_drafting_block_offsets.to(torch.long).clamp(
        min=0,
        max=num_replace_vectors - 1,
    )
    gathered_vectors = first_step_block_replace_vectors.index_select(
        0,
        safe_offsets,
    )
    gathered_vectors = gathered_vectors.to(dtype=combined_input.dtype)
    return torch.where(valid_mask, gathered_vectors, combined_input)


def get_mtp_inner_config(config: NemotronHConfig) -> NemotronHConfig:
    inner_config = copy.deepcopy(config)
    inner_config.hybrid_override_pattern = config.mtp_hybrid_override_pattern
    inner_config.num_hidden_layers = len(inner_config.hybrid_override_pattern)

    bottleneck_hidden_size = getattr(config, "mtp_bottleneck_hidden_size", None)
    if bottleneck_hidden_size is None:
        return inner_config

    inner_config.hidden_size = bottleneck_hidden_size

    def scale_ffn_size(width: int) -> int:
        denominator = config.hidden_size * 32
        return (width * bottleneck_hidden_size + denominator - 1) // denominator * 32

    inner_config.intermediate_size = scale_ffn_size(config.intermediate_size)
    inner_config.moe_intermediate_size = scale_ffn_size(config.moe_intermediate_size)
    if getattr(config, "mtp_scale_shared_expert_with_bottleneck", False):
        inner_config.moe_shared_expert_intermediate_size = scale_ffn_size(
            config.moe_shared_expert_intermediate_size
        )
    if getattr(config, "mtp_dense_mlp_match_moe_active_params", False):
        shared_width = inner_config.moe_shared_expert_intermediate_size
        if not getattr(config, "mtp_scale_shared_expert_with_bottleneck", False):
            shared_width = scale_ffn_size(shared_width)
        active_width = (
            inner_config.num_experts_per_tok * inner_config.moe_intermediate_size
            + shared_width
        )
        inner_config.intermediate_size = (active_width + 31) // 32 * 32
    return inner_config


class _NemotronHMTPDecoderLayerMixin:
    def _init_mtp_projections(
        self,
        config: NemotronHConfig,
        outer_config: NemotronHConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        has_start_projections: bool,
        has_end_norm: bool,
    ) -> None:
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm
        self.use_bottleneck_lm_head = bool(
            getattr(outer_config, "mtp_use_bottleneck_lm_head", False)
        )

        if has_start_projections:
            self.enorm = RMSNorm(
                outer_config.hidden_size, eps=outer_config.layer_norm_epsilon
            )
            self.hnorm = RMSNorm(
                outer_config.hidden_size, eps=outer_config.layer_norm_epsilon
            )
            self.eh_proj = ColumnParallelLinear(
                input_size=outer_config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=outer_config.mlp_bias,
                gather_output=True,
                params_dtype=getattr(config, "dtype", torch.bfloat16),
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            if config.hidden_size != outer_config.hidden_size:
                self.he_proj = ColumnParallelLinear(
                    input_size=config.hidden_size,
                    output_size=outer_config.hidden_size,
                    bias=outer_config.mlp_bias,
                    gather_output=True,
                    params_dtype=getattr(config, "dtype", torch.bfloat16),
                    quant_config=quant_config,
                    prefix=f"{prefix}.he_proj",
                )
            self.final_layernorm = RMSNorm(
                outer_config.hidden_size,
                eps=outer_config.layer_norm_epsilon,
            )
            if self.use_bottleneck_lm_head:
                self.bottleneck_final_layernorm = RMSNorm(
                    config.hidden_size,
                    eps=outer_config.layer_norm_epsilon,
                )

    def _apply_start_projections(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
        first_step_block_replace_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.has_start_projections:
            return hidden_states

        inputs_embeds = self.enorm(inputs_embeds)
        hidden_states = self.hnorm(hidden_states)
        fused = torch.cat([inputs_embeds, hidden_states], dim=-1)
        fused = _apply_first_step_block_replacement(
            fused,
            parallel_drafting_block_offsets,
            first_step_block_replace_vectors,
        )
        hidden_states, _ = self.eh_proj(fused)
        return hidden_states

    def _apply_end_projections(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.has_end_norm:
            return hidden_states, residual, None

        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None
        lm_head_hidden_states = None
        if self.use_bottleneck_lm_head:
            lm_head_hidden_states = self.bottleneck_final_layernorm(hidden_states)
        if hasattr(self, "he_proj"):
            hidden_states, _ = self.he_proj(hidden_states)
        return self.final_layernorm(hidden_states), residual, lm_head_hidden_states


class NemotronHMTPAttentionDecoderLayer(
    _NemotronHMTPDecoderLayerMixin, NemotronHAttentionDecoderLayer
):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        outer_config: NemotronHConfig | None = None,
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
            use_rope=bool(getattr(config, "mtp_use_rope", False)),
        )
        self._init_mtp_projections(
            config,
            outer_config or config,
            quant_config,
            prefix,
            has_start_projections,
            has_end_norm,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
        first_step_block_replace_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden_states = self._apply_start_projections(
            inputs_embeds,
            hidden_states,
            parallel_drafting_block_offsets,
            first_step_block_replace_vectors,
        )
        hidden_states, residual = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        return self._apply_end_projections(hidden_states, residual)


class NemotronHMTPMoEDecoderLayer(
    _NemotronHMTPDecoderLayerMixin, NemotronHMoEDecoderLayer
):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        outer_config: NemotronHConfig | None = None,
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self._init_mtp_projections(
            config,
            outer_config or config,
            quant_config,
            prefix,
            has_start_projections,
            has_end_norm,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
        first_step_block_replace_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden_states = self._apply_start_projections(
            inputs_embeds,
            hidden_states,
            parallel_drafting_block_offsets,
            first_step_block_replace_vectors,
        )
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return self._apply_end_projections(hidden_states, residual)


class NemotronHMTPMLPDecoderLayer(
    _NemotronHMTPDecoderLayerMixin, NemotronHMLPDecoderLayer
):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        outer_config: NemotronHConfig | None = None,
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self._init_mtp_projections(
            config,
            outer_config or config,
            quant_config,
            prefix,
            has_start_projections,
            has_end_norm,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
        first_step_block_replace_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden_states = self._apply_start_projections(
            inputs_embeds,
            hidden_states,
            parallel_drafting_block_offsets,
            first_step_block_replace_vectors,
        )
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return self._apply_end_projections(hidden_states, residual)


@support_torch_compile
class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with NemotronH layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        assert self.num_mtp_layers == 1, (
            "Only one MTP layer is supported for NemotronH-MTP"
        )

        self.pattern_str = config.mtp_hybrid_override_pattern
        self.pattern_len = len(self.pattern_str)
        assert self.pattern_len > 0
        self.naive_parallel_enabled = getattr(
            config, "mtp_naive_parallel_enabled", False
        )
        self.naive_parallel_block_len = getattr(
            config, "mtp_naive_parallel_block_len", 0
        )

        inner_config = get_mtp_inner_config(config)
        self.inner_config = inner_config

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        if self.naive_parallel_enabled:
            self.first_step_block_replace_vectors = nn.Parameter(
                torch.empty(
                    self.naive_parallel_block_len,
                    config.hidden_size * 2,
                    dtype=config.dtype if hasattr(config, "dtype") else torch.bfloat16,
                )
            )
        else:
            self.register_parameter("first_step_block_replace_vectors", None)

        # Build flat list of layers
        self.layers = torch.nn.ModuleDict()

        # Total number of physical layers = num_steps * pattern_len
        total_layers = self.num_mtp_layers * self.pattern_len
        for i in range(total_layers):
            step_rel_idx = i % self.pattern_len

            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            layer_prefix = f"{prefix}.layers.{i}"
            layer_config = copy.deepcopy(inner_config)
            layer_config.sliding_window = (
                config.mtp_window_size[0] if char == "W" else None
            )
            layer_config.mtp_attention_softmax_type = (
                getattr(config, "mtp_softmax_type", "vanilla")
                if char in ("*", "W")
                else "vanilla"
            )

            # TODO smor- remove double layers formation
            common_kwargs = dict(
                config=layer_config,
                outer_config=config,
                layer_idx=step_rel_idx,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                parallel_config=vllm_config.parallel_config,
                prefix=layer_prefix,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
            )

            if char in ("*", "W"):
                self.layers[str(i)] = NemotronHMTPAttentionDecoderLayer(**common_kwargs)
            elif char == "E":
                self.layers[str(i)] = NemotronHMTPMoEDecoderLayer(**common_kwargs)
            elif char == "-":
                self.layers[str(i)] = NemotronHMTPMLPDecoderLayer(**common_kwargs)
            else:
                raise NotImplementedError(
                    f"Pattern char '{char}' in {self.pattern_str} not implemented"
                )

        self.make_empty_intermediate_tensors: Callable[..., IntermediateTensors] = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        residual = None
        lm_head_hidden_states = None

        for i in range(self.pattern_len):
            hidden_states, residual, layer_lm_head_hidden_states = self.layers[str(i)](
                inputs_embeds=inputs_embeds,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                parallel_drafting_block_offsets=parallel_drafting_block_offsets,
                first_step_block_replace_vectors=(
                    self.first_step_block_replace_vectors
                ),
            )
            if layer_lm_head_hidden_states is not None:
                lm_head_hidden_states = layer_lm_head_hidden_states
        if getattr(self.config, "mtp_use_bottleneck_lm_head", False):
            assert lm_head_hidden_states is not None
            # MRV2 uses the first tensor for logits and feeds the second back
            # into the next autoregressive MTP step.
            return lm_head_hidden_states, hidden_states
        return hidden_states


class NemotronHMTP(nn.Module, SupportsPP):
    """NemotronH MTP model."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        # MRV2 normally aliases the draft head to the target head. U configs
        # export a separately trained bottleneck-width head, so preserve it.
        self.has_own_lm_head = bool(
            getattr(config, "mtp_use_bottleneck_lm_head", False)
        )

        # Needed for load_weights mapping
        self.mtp_start_layer_idx = config.num_hidden_layers

        # EPLB config for experts
        self.num_redundant_experts = 0
        if vllm_config.parallel_config and vllm_config.parallel_config.eplb_config:
            self.num_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )

        # MTP predictor
        self.model = NemotronHMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            (
                self.config.mtp_bottleneck_hidden_size
                if self.has_own_lm_head
                else self.config.hidden_size
            ),
            prefix=maybe_prefix(
                prefix,
                "mtp.output_layer"
                if self.has_own_lm_head
                else "lm_head",
            ),
        )

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        parallel_drafting_block_offsets: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward - applies attention-based MTP."""
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            parallel_drafting_block_offsets,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = []
        num_experts = getattr(self.config, "n_routed_experts", None)
        if getattr(self.config, "model_type", None) == "nemotron_h_puzzle":
            num_experts = self.config.mtp_n_routed_experts
        if num_experts is not None:
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",  # Empty - non-gated MoE
                num_experts=num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unconsumed_mtp_weights: set[str] = set()

        for name, loaded_weight in weights:
            checkpoint_name = name
            # Only process MTP weights - skip all non-MTP weights
            if not name.startswith("mtp.") and "embeddings" not in name:
                continue
            # Skip rotary embeddings (computed, not loaded)
            if "rotary_emb.inv_freq" in name:
                continue
            if checkpoint_name.startswith("mtp."):
                unconsumed_mtp_weights.add(checkpoint_name)

            if name == "mtp.first_step_block_replace_vectors":
                name = "model.first_step_block_replace_vectors"
            elif name == "mtp.output_layer.weight":
                name = "lm_head.weight"
            else:
                name = name.replace("mtp.layers.", "model.layers.")

            if "embeddings" in name:
                name = name.replace("embeddings", "embed_tokens")
                if name.startswith("backbone."):
                    name = name.replace("backbone.", "model.")

            # Handle stacked parameters (qkv_proj) for attention layers
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Must be in a mixer (attention layer)
                if ".mixer." not in name:
                    continue

                is_stacked = True
                stacked_name = name.replace(weight_name, param_name)

                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue

                if stacked_name not in params_dict:
                    # Might be that mapping failed or param doesn't exist
                    continue

                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(stacked_name)
                    unconsumed_mtp_weights.discard(checkpoint_name)
                break

            if is_stacked:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                # weight_name is like "experts.0.up_proj."
                if weight_name not in name:
                    continue

                is_expert_weight = True

                # Replace the expert-specific weight name with fused parameter name
                # e.g., "experts.0.up_proj." -> "experts.w13_"
                name_mapped = name.replace(weight_name, param_name)

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                    unconsumed_mtp_weights.discard(checkpoint_name)
                break

            if is_expert_weight:
                # Expert-parallel ranks receive the full checkpoint iterator but
                # only materialize their local experts. Reaching this branch means
                # the tensor matched a known expert mapping and belongs elsewhere.
                unconsumed_mtp_weights.discard(checkpoint_name)
                continue

            if name.endswith(".mixer.sinks"):
                if name not in params_dict:
                    continue
                param = params_dict[name]
                sharded_weight_loader(0)(param, loaded_weight)
                loaded_params.add(name)
                unconsumed_mtp_weights.discard(checkpoint_name)
                continue

            if name.endswith(".bias") and name not in params_dict:
                unconsumed_mtp_weights.discard(checkpoint_name)
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            unconsumed_mtp_weights.discard(checkpoint_name)

        if unconsumed_mtp_weights:
            raise ValueError(
                "Nemotron-H MTP checkpoint tensors were not consumed: "
                f"{sorted(unconsumed_mtp_weights)[:20]}"
            )

        required_params = set()
        if self.has_own_lm_head:
            required_params.add("lm_head.weight")
            required_params.add(
                f"model.layers.{self.model.pattern_len - 1}."
                "bottleneck_final_layernorm.weight"
            )
        if getattr(self.config, "mtp_softmax_type", "vanilla") == "learnable":
            for layer_idx, symbol in enumerate(self.model.pattern_str):
                if symbol in ("W", "*"):
                    required_params.add(f"model.layers.{layer_idx}.mixer.sinks")
        missing_required = required_params - loaded_params
        if missing_required:
            raise ValueError(
                "Nemotron-H MTP feature tensors were not loaded: "
                f"{sorted(missing_required)}"
            )

        if (
            getattr(self.config, "mtp_naive_parallel_enabled", False)
            and "model.first_step_block_replace_vectors" not in loaded_params
        ):
            raise ValueError(
                "Checkpoint config enables Nemotron-H MTP parallel drafting, but "
                "`mtp.first_step_block_replace_vectors` was not found in the "
                "loaded weights."
            )

        return loaded_params
