# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4W4A16LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def _config(*, method: str, share: bool) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = method
    config.num_speculative_tokens = 1
    config.mtp_share_lm_head = share
    config.draft_model_config = None
    return config


def _proposer(*, share: bool, draft_head: nn.Module) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.speculative_config = SimpleNamespace(
        mtp_share_lm_head=share,
        draft_model_config=SimpleNamespace(model="draft-checkpoint"),
    )
    proposer.model = nn.Module()
    proposer.model.lm_head = draft_head
    proposer.model.model = nn.Module()
    proposer.use_local_argmax_reduction = False
    proposer.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace())
        )
    )
    return proposer


def _loader_model(*, private: bool, share_embeddings: bool = True) -> NemotronHMTP:
    model = object.__new__(NemotronHMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=None, model_type="nemotron_h")
    model.num_redundant_experts = 0
    model.use_private_lm_head = private
    model.share_target_embeddings = share_embeddings
    return model


def test_private_lm_head_is_rejected_for_non_mtp() -> None:
    with pytest.raises(ValueError, match="only supported for MTP"):
        _config(method="draft_model", share=False)._verify_args()


def test_mtp_share_lm_head_affects_config_hash() -> None:
    shared = _config(method="mtp", share=True)
    private = _config(method="mtp", share=False)
    assert shared.compute_hash() != private.compute_hash()


def test_mtp_shares_lm_head_by_default() -> None:
    target = nn.Module()
    target.lm_head = nn.Linear(2, 3, bias=False)
    target.model = nn.Module()
    proposer = _proposer(share=True, draft_head=nn.Linear(2, 3, bias=False))
    proposer._maybe_share_lm_head(target)
    assert proposer.model.lm_head is target.lm_head


def test_mtp_private_lm_head_has_distinct_storage_and_logits() -> None:
    target = nn.Module()
    target.lm_head = nn.Linear(2, 3, bias=False)
    target.model = nn.Module()
    draft_head = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        target.lm_head.weight.fill_(1.0)
        draft_head.weight.fill_(2.0)
    proposer = _proposer(share=False, draft_head=draft_head)

    proposer._maybe_share_lm_head(target)

    assert proposer.model.lm_head is draft_head
    assert (
        proposer.model.lm_head.weight.untyped_storage().data_ptr()
        != target.lm_head.weight.untyped_storage().data_ptr()
    )
    hidden_states = torch.ones(1, 2)
    assert not torch.equal(
        target.lm_head(hidden_states), proposer.model.lm_head(hidden_states)
    )


def test_private_mtp_loads_all_three_w4a16_head_tensors() -> None:
    model = _loader_model(private=True)
    model.lm_head = nn.Module()
    model.lm_head.weight = nn.Parameter(torch.zeros(4, 2))
    model.lm_head.weight_scale = nn.Parameter(torch.zeros(4, 1))
    model.lm_head.weight_scale_2 = nn.Parameter(torch.zeros(1))
    weights = [
        ("lm_head.weight", torch.ones(4, 2)),
        ("lm_head.weight_scale", torch.full((4, 1), 0.5)),
        ("lm_head.weight_scale_2", torch.tensor([0.25])),
    ]

    loaded = model.load_weights(weights)

    assert loaded == {
        "lm_head.weight",
        "lm_head.weight_scale",
        "lm_head.weight_scale_2",
    }
    assert torch.equal(model.lm_head.weight, weights[0][1])
    assert torch.equal(model.lm_head.weight_scale, weights[1][1])
    assert torch.equal(model.lm_head.weight_scale_2, weights[2][1])


def test_private_mtp_rejects_checkpoint_without_lm_head_weights() -> None:
    model = _loader_model(private=True)
    with pytest.raises(ValueError, match="requires lm_head weights"):
        model.load_weights([])


def test_quantized_mtp_rejects_any_uninitialized_parameter() -> None:
    model = _loader_model(private=True)
    model.lm_head = nn.Module()
    model.lm_head.weight = nn.Parameter(torch.zeros(3, 2))
    model.unmapped_weight = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="did not initialize 1 parameter"):
        model.load_weights([("lm_head.weight", torch.ones(3, 2))])


def test_quantized_mtp_rejects_partial_fused_qkv_loading() -> None:
    model = _loader_model(private=False, share_embeddings=False)
    model.lm_head = nn.Linear(1, 1, bias=False)
    model.model = nn.Module()
    model.model.layers = nn.ModuleDict({"0": nn.Module()})
    model.model.layers["0"].mixer = nn.Module()
    qkv_weight = nn.Parameter(torch.zeros(3, 1))

    def load_qkv_shard(param, loaded_weight, shard_id):
        param.data[{"q": 0, "k": 1, "v": 2}[shard_id]].copy_(loaded_weight)

    qkv_weight.weight_loader = load_qkv_shard
    model.model.layers["0"].mixer.qkv_proj = nn.Module()
    model.model.layers["0"].mixer.qkv_proj.weight = qkv_weight
    with pytest.raises(ValueError, match="incompletely initialized fused QKV"):
        model.load_weights(
            [("mtp.layers.0.mixer.q_proj.weight", torch.ones(1))]
        )


def test_quantized_mtp_rejects_partial_fused_expert_loading() -> None:
    model = _loader_model(private=False)
    model.config.n_routed_experts = 2
    model.lm_head = nn.Linear(1, 1, bias=False)
    model.model = nn.Module()
    model.model.layers = nn.ModuleDict({"0": nn.Module()})
    model.model.layers["0"].mixer = nn.Module()
    model.model.layers["0"].mixer.experts = nn.Module()
    fused_weight = nn.Parameter(torch.zeros(2, 1))

    def load_one_expert(
        param, loaded_weight, name, *, shard_id, expert_id, return_success
    ):
        if expert_id != 0:
            return False
        param.data[expert_id].copy_(loaded_weight)
        return True

    fused_weight.weight_loader = load_one_expert
    model.model.layers["0"].mixer.experts.w13_weight = fused_weight
    mapping = [
        ("experts.w13_", "experts.0.up_proj.", 0, "w1"),
        ("experts.w13_", "experts.1.up_proj.", 1, "w1"),
    ]
    weights = [
        ("mtp.layers.0.mixer.experts.0.up_proj.weight", torch.ones(1)),
        ("mtp.layers.0.mixer.experts.1.up_proj.weight", torch.ones(1)),
    ]
    with (
        mock.patch(
            "vllm.model_executor.models.nemotron_h_mtp."
            "fused_moe_make_expert_params_mapping",
            return_value=mapping,
        ),
        pytest.raises(ValueError, match=r"1/2 expert slices"),
    ):
        model.load_weights(weights)


@pytest.mark.parametrize("share_lm_head", [True, False])
def test_mtp_uses_draft_not_target_quant_config(share_lm_head: bool) -> None:
    @dataclass
    class AttentionConfig:
        backend: object = None

    @dataclass
    class FakeVllmConfig:
        model_config: object
        quant_config: object
        load_config: object
        attention_config: AttentionConfig
        kernel_config: object = None

    target_quant_config = object()
    draft_quant_config = object()
    draft_model_config = object()
    base = FakeVllmConfig(
        model_config=object(),
        quant_config=target_quant_config,
        load_config=object(),
        attention_config=AttentionConfig(),
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.vllm_config = base
    proposer.speculative_config = SimpleNamespace(
        mtp_share_lm_head=share_lm_head,
        draft_model_config=draft_model_config,
        draft_load_config=None,
        moe_backend=None,
        attention_backend=None,
        kv_cache_dtype=None,
    )
    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer."
        "VllmConfig._get_quantization_config",
        return_value=draft_quant_config,
    ) as get_quant_config:
        result = proposer._create_draft_vllm_config()

    get_quant_config.assert_called_once_with(draft_model_config, base.load_config)
    assert result.quant_config is draft_quant_config
    assert result.quant_config is not target_quant_config
    assert result.model_config is draft_model_config


def test_shared_mtp_allows_target_owned_weights_to_be_absent() -> None:
    model = _loader_model(private=False)
    model.lm_head = nn.Module()
    model.lm_head.weight = nn.Parameter(torch.zeros(3, 2))
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(3, 2)
    loaded = model.load_weights([])
    assert "lm_head.weight" in loaded
    assert "model.embed_tokens.weight" in loaded


@pytest.mark.parametrize(
    ("share", "expected_quant_config"),
    [(True, None), (False, "draft")],
)
def test_nemotron_mtp_private_head_gets_draft_quant_config(
    share: bool, expected_quant_config: str | None
) -> None:
    class FakePredictor(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.make_empty_intermediate_tensors = mock.MagicMock()

    quant_config = mock.MagicMock()
    quant_config.get_name.return_value = "modelopt"
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=8, hidden_size=4, num_hidden_layers=1
            )
        ),
        parallel_config=None,
        quant_config=quant_config,
        speculative_config=SimpleNamespace(
            method="mtp", mtp_share_lm_head=share
        ),
    )
    with (
        mock.patch(
            "vllm.model_executor.models.nemotron_h_mtp."
            "NemotronHMultiTokenPredictor",
            FakePredictor,
        ),
        mock.patch(
            "vllm.model_executor.models.nemotron_h_mtp.ParallelLMHead"
        ) as lm_head_cls,
        mock.patch(
            "vllm.model_executor.models.nemotron_h_mtp.LogitsProcessor"
        ),
    ):
        NemotronHMTP(vllm_config=vllm_config)

    expected = quant_config if expected_quant_config == "draft" else None
    assert lm_head_cls.call_args.kwargs["quant_config"] is expected


def test_modelopt_w4a16_constructs_packed_parallel_lm_head() -> None:
    w4a16_config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
    )
    quant_method = object.__new__(ModelOptNvFp4W4A16LinearMethod)
    quant_method.quant_config = w4a16_config
    mixed_config = ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers={
            "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        },
        fp8_config=mock.MagicMock(),
        nvfp4_config=mock.MagicMock(),
        w4a16_nvfp4_config=w4a16_config,
        mxfp8_config=mock.MagicMock(),
    )
    with (
        mock.patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptNvFp4W4A16LinearMethod",
            return_value=quant_method,
        ),
        mock.patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        head = ParallelLMHead(
            num_embeddings=8,
            embedding_dim=32,
            quant_config=mixed_config,
            prefix="draft.lm_head",
        )

    assert head.quant_config is mixed_config
    assert head.quant_method is quant_method
    assert tuple(head.weight.shape) == (64, 16)
    assert tuple(head.weight_scale.shape) == (64, 2)
    assert tuple(head.weight_scale_2.shape) == (1,)
