"""CUDA sampling startup checks without model weights or HTTP requests.

Run with ``python -m sglang.test.sampling_prewarm --vocab-size N``. Each
phase uses a fresh process and a private cache, leaving serving caches alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def cache_environment(root: Path) -> dict[str, str]:
    return {
        "SGLANG_CACHE_DIR": str(root),
        "SGLANG_JIT_CACHE_DIR": str(root / "jit"),
        "FLASHINFER_WORKSPACE_BASE": str(root),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "CUDA_CACHE_PATH": str(root / "nv"),
    }


def build_artifacts(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in (".o", ".so") or path.name == ".ninja_log")
    }


def age_sampling_artifacts(root: Path) -> int:
    """Invalidate only this check's FlashInfer outputs, never installed sources."""
    paths = [
        path
        for path in root.glob(".cache/flashinfer/*/*/cached_ops/sampling/*")
        if path.is_file() and path.suffix in (".o", ".so")
    ]
    for path in paths:
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    return len(paths)


def check_cuda_sampling(vocab_size: int) -> dict:
    import importlib.metadata
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("This check requires an NVIDIA CUDA GPU")

    import flashinfer.sampling as sampling

    from sglang.kernels.ops.speculative import sampling as tree
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.is_generation = True
    runner.is_draft_worker = False
    runner.model_config = SimpleNamespace(vocab_size=vocab_size)
    durations = {}
    # Invoke the actual startup hook. No weights, scheduler loop or model
    # request is needed to build/load these shape-independent modules.
    with get_context().override_server_args(
        sampling_backend="flashinfer", speculative_use_rejection_sampling=False
    ):
        for algorithm in (SpeculativeAlgorithm.NONE, SpeculativeAlgorithm.EAGLE):
            runner.spec_algorithm = algorithm
            started = time.perf_counter()
            runner.prewarm_sampling()
            durations[algorithm.name] = time.perf_counter() - started

    probs = torch.zeros((1, vocab_size), device="cuda", dtype=torch.float32)
    probs[0, 1] = 1.0
    # The original upstream preflight used (1, 1). FlashInfer 0.6.18
    # requires (batch_size,), so exercise real operators with 1D inputs.
    top_ks = torch.tensor([2], device="cuda", dtype=torch.int32)
    top_ps = torch.tensor([0.95], device="cuda", dtype=torch.float32)
    late_build = AssertionError(
        "Sampling attempted to build/load after startup prewarm"
    )
    started = time.perf_counter()
    # Only the cold builder is replaced. Real cached modules and GPU kernels
    # still run; a missed startup path fails instead of compiling silently.
    with (
        patch.object(sampling, "gen_sampling_module", side_effect=late_build),
        patch.object(tree, "load_jit", side_effect=late_build),
    ):
        for _ in range(2):
            torch.testing.assert_close(
                sampling.top_k_renorm_probs(probs, top_ks), probs
            )
            torch.testing.assert_close(
                sampling.top_p_renorm_probs(probs, top_ps), probs
            )
            token = sampling.top_k_top_p_sampling_from_probs(
                probs, top_ks, top_ps, filter_apply_order="joint"
            )
            assert token.item() == 1
            token = sampling.min_p_sampling_from_probs(probs, 0.1)
            assert token.item() == 1

            candidates = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
            predicts = torch.full((2,), -1, dtype=torch.int32, device="cuda")
            accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
            accept_count = torch.zeros((1,), dtype=torch.int32, device="cuda")
            target_probs = probs[:, None, :].expand(1, 2, vocab_size).contiguous()
            tree.tree_speculative_sampling_target_only(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_count,
                candidates=candidates,
                retrive_index=candidates.clone(),
                retrive_next_token=torch.tensor(
                    [[1, -1]], dtype=torch.int64, device="cuda"
                ),
                retrive_next_sibling=torch.full_like(candidates, -1),
                uniform_samples=torch.zeros((1, 2), device="cuda"),
                uniform_samples_for_final_sampling=torch.zeros((1,), device="cuda"),
                target_probs=target_probs,
                draft_probs=torch.zeros_like(target_probs),
                deterministic=True,
            )
            assert predicts.tolist() == [1, 1]
            assert accept_count.item() == 1
            assert accept_index.tolist() == [[0, 1]]
        torch.cuda.synchronize()
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "vocab_size": vocab_size,
        "startup_seconds": durations,
        "sampling_seconds": time.perf_counter() - started,
        "no_late_build": True,
        "kernel_outputs": "passed",
    }


def run_checks(output: Path, vocab_size: int, timeout: int) -> dict:
    cache = output / "cache"
    env = dict(os.environ, **cache_environment(cache))
    results = {}

    def phase(name: str):
        result_path = output / f"{name}.json"
        try:
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "sglang.test.sampling_prewarm",
                        "--worker",
                        "--vocab-size",
                        str(vocab_size),
                        "--output-dir",
                        str(result_path),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            log_path = output / f"{name}.log"
            print(f"{name} failed; log: {log_path}", file=sys.stderr)
            print("\n".join(log_path.read_text().splitlines()[-80:]), file=sys.stderr)
            raise
        results[name] = json.loads(result_path.read_text())
        print(f"{name}: kernel checks passed", flush=True)

    phase("cold")
    before = build_artifacts(cache)
    phase("warm")
    if build_artifacts(cache) != before:
        raise AssertionError("Warm-cache startup changed compiled artifacts")
    aged = age_sampling_artifacts(cache)
    if aged:
        before = build_artifacts(cache)
        phase("stale")
        if build_artifacts(cache) == before:
            raise AssertionError("Stale sampling artifacts were not rebuilt")
        before = build_artifacts(cache)
        phase("warm_after_stale")
        if build_artifacts(cache) != before:
            raise AssertionError(
                "Warm startup after rebuild changed compiled artifacts"
            )
    else:
        results["stale"] = {
            "status": "not_exercised",
            "reason": "No FlashInfer JIT outputs; check whether this image uses prebuilt kernels",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Seconds per CUDA subprocess"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vocab_size < 4 or args.timeout <= 0:
        parser.error("vocab-size must be >= 4 and timeout must be positive")
    if args.worker:
        args.output_dir.write_text(
            json.dumps(check_cuda_sampling(args.vocab_size), indent=2) + "\n"
        )
        return
    if args.output_dir is None:
        output = Path(tempfile.mkdtemp(prefix="sampling-prewarm-"))
    else:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence directory: {output}", flush=True)
    try:
        results = run_checks(output, args.vocab_size, args.timeout)
    except Exception as error:
        (output / "FAILED.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        "CUDA sampling checks passed; service readiness and HTTP checks are separate."
    )


if __name__ == "__main__":
    main()
