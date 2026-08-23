# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestOutputCollector


def _request_output(token_ids: list[int], step_counts: list[int]) -> RequestOutput:
    return RequestOutput(
        request_id="request",
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=None,
                scheduler_step_token_counts=step_counts,
            )
        ],
        finished=False,
    )


def test_delta_collector_preserves_coalesced_scheduler_steps() -> None:
    collector = RequestOutputCollector(RequestOutputKind.DELTA, "request")
    collector.put(_request_output([10, 11, 12, 13], [4]))
    collector.put(_request_output([14], [1]))
    collector.put(_request_output([15, 16, 17], [3]))

    output = collector.get_nowait()
    assert output is not None
    completion = output.outputs[0]
    assert completion.token_ids == [10, 11, 12, 13, 14, 15, 16, 17]
    assert completion.scheduler_step_token_counts == [4, 1, 3]


def test_cumulative_collector_preserves_replaced_scheduler_steps() -> None:
    collector = RequestOutputCollector(RequestOutputKind.CUMULATIVE, "request")
    collector.put(_request_output([10, 11, 12, 13], [4]))
    collector.put(_request_output([10, 11, 12, 13, 14], [1]))
    collector.put(_request_output([10, 11, 12, 13, 14, 15, 16, 17], [3]))

    output = collector.get_nowait()
    assert output is not None
    completion = output.outputs[0]
    assert completion.token_ids == [10, 11, 12, 13, 14, 15, 16, 17]
    assert completion.scheduler_step_token_counts == [4, 1, 3]
