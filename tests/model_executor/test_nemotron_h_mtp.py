# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHMTP,
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


def test_mtp_inner_config_scales_shared_experts():
    config = NemotronHConfig(
        hidden_size=2688,
        intermediate_size=1856,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        num_experts_per_tok=6,
        num_hidden_layers=1,
        hybrid_override_pattern="E",
        mtp_hybrid_override_pattern="E",
        num_nextn_predict_layers=1,
        mtp_bottleneck_hidden_size=1024,
        mtp_scale_shared_expert_with_bottleneck=True,
    )

    inner_config = get_mtp_inner_config(config)

    assert inner_config.moe_intermediate_size == 736
    assert inner_config.moe_shared_expert_intermediate_size == 1440


def test_mtp_inner_config_matches_active_moe_width_for_dense_layers():
    config = NemotronHConfig(
        hidden_size=2688,
        intermediate_size=1856,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        num_experts_per_tok=6,
        num_hidden_layers=1,
        hybrid_override_pattern="-",
        mtp_hybrid_override_pattern="-",
        num_nextn_predict_layers=1,
        mtp_bottleneck_hidden_size=1024,
        mtp_dense_mlp_match_moe_active_params=True,
    )

    inner_config = get_mtp_inner_config(config)

    assert inner_config.intermediate_size == 5856


def test_mtp_loader_requires_configured_feature_weights():
    model = NemotronHMTP.__new__(NemotronHMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        model_type="nemotron_h",
        n_routed_experts=None,
        mtp_naive_parallel_enabled=False,
        mtp_use_bottleneck_lm_head=True,
        mtp_softmax_type="learnable",
    )
    model.num_redundant_experts = 0
    model.model = SimpleNamespace(pattern_len=2, pattern_str="W*")

    with pytest.raises(ValueError, match="feature tensors were not loaded"):
        model.load_weights([])
