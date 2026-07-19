# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def _create_mtp_proposer(
    num_speculative_tokens: int,
    parallel_drafting: bool = False,
    naive_parallel_block_len: int | None = None,
) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=parallel_drafting,
    )
    if parallel_drafting:
        hf_config = speculative_config.draft_model_config.hf_config
        hf_config.model_type = "nemotron_h_mtp"
        hf_config.__dict__.pop("ptd_token_id", None)
        hf_config.pad_token_id = 7
        hf_config.mtp_naive_parallel_enabled = True
        hf_config.mtp_naive_parallel_block_len = (
            num_speculative_tokens - 1
            if naive_parallel_block_len is None
            else naive_parallel_block_len
        )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    proposer = EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)
    # Production initializes this when the KV cache is attached. Unit tests
    # exercise propose() directly, so mirror that initialization here.
    proposer.block_size = 16
    return proposer


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)


def test_mtp_parallel_drafting_rejects_more_masked_slots_than_trained():
    draft_hf_config = SimpleNamespace(
        model_type="nemotron_h_mtp",
        mtp_naive_parallel_enabled=True,
        mtp_naive_parallel_block_len=2,
    )
    draft_model_config = mock.MagicMock()
    draft_model_config.hf_config = draft_hf_config
    draft_model_config.get_vocab_size.return_value = 100

    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.num_speculative_tokens = 4
    speculative_config.method = "mtp"
    speculative_config.parallel_drafting = True
    speculative_config.draft_model_config = draft_model_config

    with pytest.raises(
        ValueError,
        match="requires 3 masked slots, but the checkpoint was trained for at most 2",
    ):
        speculative_config._verify_parallel_drafting_mtp()


def test_mtp_parallel_drafting_passes_block_offsets_to_model():
    device = torch.device(current_platform.device_type)
    batch_size = 2
    seq_lens = [4, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100
    num_speculative_tokens = 4

    proposer = _create_mtp_proposer(
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=True,
    )
    hidden_size = proposer.hidden_size
    assert proposer.parallel_drafting_token_id == 7

    def mock_forward(**kwargs):
        num_tokens = kwargs["positions"].shape[0]
        return torch.zeros(num_tokens, hidden_size, device=device)

    model_mock = mock.MagicMock(side_effect=mock_forward)
    model_mock.compute_logits.return_value = torch.zeros(
        batch_size * num_speculative_tokens, vocab_size, device=device
    )
    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(
        total_tokens, hidden_size, dtype=proposer.dtype, device=device
    )
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    assert result.shape == (batch_size, num_speculative_tokens)
    kwargs = model_mock.call_args.kwargs
    assert "parallel_drafting_block_offsets" in kwargs
    expected_offsets = torch.tensor(
        [-1, -1, -1, -1, 0, 1, 2, -1, -1, -1, 0, 1, 2],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(kwargs["parallel_drafting_block_offsets"], expected_offsets)


def test_mtp_parallel_drafting_allocates_offsets_for_single_token():
    proposer = _create_mtp_proposer(
        num_speculative_tokens=1,
        parallel_drafting=True,
        naive_parallel_block_len=8,
    )

    assert proposer.parallel_drafting_uses_block_offsets is True
    assert proposer.needs_extra_input_slots is False
    assert proposer.parallel_drafting_block_offsets is not None


def test_mtp_parallel_drafting_dummy_run_passes_block_offsets(monkeypatch):
    device = torch.device(current_platform.device_type)
    num_tokens = 5
    padded_num_tokens = 8
    proposer = _create_mtp_proposer(
        num_speculative_tokens=4,
        parallel_drafting=True,
    )
    proposer.model = mock.MagicMock()
    proposer._draft_attn_layer_names = set()

    monkeypatch.setattr(
        proposer,
        "_determine_batch_execution_and_padding",
        lambda num_tokens, use_cudagraphs=True: (
            CUDAGraphMode.NONE,
            padded_num_tokens,
            None,
        ),
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.llm_base_proposer.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    proposer.dummy_run(
        num_tokens,
        use_cudagraphs=False,
        is_graph_capturing=True,
    )

    kwargs = proposer.model.call_args.kwargs
    assert "parallel_drafting_block_offsets" in kwargs
    assert torch.equal(
        kwargs["parallel_drafting_block_offsets"],
        torch.full((padded_num_tokens,), -1, dtype=torch.int32, device=device),
    )


def test_mtp_parallel_drafting_clears_rejected_and_padded_rows(monkeypatch):
    device = torch.device(current_platform.device_type)
    batch_size = 2
    seq_lens = [4, 4]
    total_tokens = sum(seq_lens)
    vocab_size = 100
    num_speculative_tokens = 3

    proposer = _create_mtp_proposer(
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=True,
    )
    hidden_size = proposer.hidden_size

    padded_num_tokens = 16

    def mock_forward(**kwargs):
        offsets = kwargs["parallel_drafting_block_offsets"]
        hidden_states = kwargs["hidden_states"]
        input_ids = kwargs["input_ids"]

        assert torch.equal(
            offsets[12:padded_num_tokens],
            torch.full(
                (padded_num_tokens - 12,),
                -1,
                dtype=torch.int32,
                device=device,
            ),
        )
        assert torch.equal(
            hidden_states[5],
            torch.zeros(hidden_size, dtype=hidden_states.dtype, device=device),
        )
        assert torch.equal(
            hidden_states[12:padded_num_tokens],
            torch.zeros(
                (padded_num_tokens - 12, hidden_size),
                dtype=hidden_states.dtype,
                device=device,
            ),
        )
        assert torch.equal(
            input_ids[12:padded_num_tokens],
            torch.full(
                (padded_num_tokens - 12,),
                proposer.parallel_drafting_token_id,
                dtype=torch.int32,
                device=device,
            ),
        )
        return torch.zeros(padded_num_tokens, hidden_size, device=device)

    model_mock = mock.MagicMock(side_effect=mock_forward)
    model_mock.compute_logits.return_value = torch.zeros(
        batch_size * num_speculative_tokens, vocab_size, device=device
    )
    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.tensor(
        [10, 11, 12, 13, 20, 21, 22, 23],
        dtype=torch.int32,
        device=device,
    )
    target_positions = torch.tensor(
        [5, 6, 7, 8, 10, 11, 12, 13],
        dtype=torch.int64,
        device=device,
    )
    target_hidden_states = torch.arange(
        total_tokens * hidden_size,
        dtype=proposer.dtype,
        device=device,
    ).view(total_tokens, hidden_size)
    next_token_ids = torch.tensor([100, 200], dtype=torch.int32, device=device)
    num_rejected_tokens_gpu = torch.tensor([1, 0], dtype=torch.int32, device=device)
    sampling_metadata = mock.MagicMock()

    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]
    monkeypatch.setattr(
        proposer,
        "_determine_batch_execution_and_padding",
        lambda num_tokens: (CUDAGraphMode.FULL, padded_num_tokens, None),
    )

    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )

    assert result.shape == (batch_size, num_speculative_tokens)
