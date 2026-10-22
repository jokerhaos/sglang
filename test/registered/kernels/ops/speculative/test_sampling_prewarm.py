"""Exercise startup JIT and real sampling in fresh, isolated CUDA processes."""

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sampling_prewarm import run_checks

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_sampling_prewarm(tmp_path):
    results = run_checks(tmp_path, vocab_size=128, timeout=1800)
    assert results["cold"]["no_late_build"]
    assert results["warm"]["no_late_build"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
