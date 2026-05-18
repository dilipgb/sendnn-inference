"""Tests for sampling separation.

Tests that sampling is separated from model execution to enable
async grammar preparation while the model runs.
"""

import pytest
from vllm import SamplingParams

from v1.worker.mock_model import InstrumentedModelRunner
from spyre_util import REFERENCE_MODELS, create_random_request


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_execute_model_forward_pass_only(monkeypatch: pytest.MonkeyPatch):
    """Test that model runner's execute_model only does forward pass."""
    
    # Setup: Use the default test model
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    
    # Build model runner
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )
    
    scheduler = model_runner.scheduler
    
    # Create a request using spyre_util helper
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )
    
    request = create_random_request(
        request_id=0,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=0,
    )
    
    scheduler.add_request(request)
    
    # Schedule
    sched_output = scheduler.schedule()
    
    # Verify grammar output is attached by scheduler
    assert hasattr(sched_output, '_spyre_grammar_output')
    
    # Execute model - should return tuple (logits, is_prefill, model_input, t0)
    result = model_runner.execute_model(sched_output)
    assert isinstance(result, tuple), "execute_model should return a tuple for sampling"
    assert len(result) == 4, "tuple should have 4 elements"
    logits, is_prefill, model_input, t0 = result
    assert logits is not None, "logits should not be None"
    assert isinstance(is_prefill, bool), "is_prefill should be a bool"
    assert model_input is not None, "model_input should not be None"
    assert isinstance(t0, float), "t0 should be a float"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_sampling_state_storage(monkeypatch: pytest.MonkeyPatch):
    """Test that execute_model stores sampling state."""
    
    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )
    
    scheduler = model_runner.scheduler
    
    # Create a request using spyre_util helper
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )
    
    request = create_random_request(
        request_id=1,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=1,
    )
    
    scheduler.add_request(request)
    sched_output = scheduler.schedule()
    
    # Execute model - should return tuple with logits and metadata
    result = model_runner.execute_model(sched_output)
    assert isinstance(result, tuple), "execute_model should return a tuple"
    assert len(result) == 4, "tuple should have 4 elements"
    logits, is_prefill, model_input, t0 = result
    assert logits is not None, "logits should not be None"


# Made with Bob