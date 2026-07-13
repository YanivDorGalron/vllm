# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.nemotron_h_mtp import get_mtp_inner_config
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
