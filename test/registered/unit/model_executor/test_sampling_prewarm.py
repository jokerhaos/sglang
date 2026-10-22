"""Sampling JIT must finish during worker startup, before serving requests."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test import sampling_prewarm as validation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.speculative import sampling as tree_sampling
from sglang.srt.model_executor.model_runner import ModelRunner, SamplingPrewarmResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RUNNER_MODULE = "sglang.srt.model_executor.model_runner"


class TestSamplingPrewarm(CustomTestCase):
    def setUp(self):
        self.loader = Mock()
        self.tree_loader = Mock()
        sampling = ModuleType("flashinfer.sampling")
        sampling.get_sampling_module = self.loader
        self.patches = (
            patch.dict(sys.modules, {"flashinfer.sampling": sampling}),
            patch(f"{_RUNNER_MODULE}.is_cuda", return_value=True),
            patch.object(
                tree_sampling, "prewarm_tree_speculative_sampling", self.tree_loader
            ),
            patch(
                f"{_RUNNER_MODULE}.get_spec",
                return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
            ),
            patch(
                f"{_RUNNER_MODULE}.get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="flashinfer")
                ),
            ),
        )
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _runner(self, *, algorithm="NONE", **overrides):
        runner = ModelRunner.__new__(ModelRunner)
        runner.device = "cuda"
        runner.is_generation = True
        runner.is_draft_worker = False
        runner.model_config = SimpleNamespace(vocab_size=129280)
        runner.spec_algorithm = SpeculativeAlgorithm.from_string(algorithm)
        runner.__dict__.update(overrides)
        return runner

    def test_prefill_builds_and_loads_sampling_before_returning(self):
        runner = self._runner()
        result = runner.prewarm_sampling()
        self.loader.assert_called_once_with()
        self.tree_loader.assert_not_called()
        self.assertEqual(result, SamplingPrewarmResult())
        self.assertIs(runner.sampling_prewarm_result, result)

    def test_eagle_delegates_to_target_even_with_pytorch_sampler(self):
        for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            with self.subTest(algorithm=algorithm):
                self.loader.reset_mock()
                self.tree_loader.reset_mock()
                runner = self._runner(algorithm=algorithm)
                worker = BaseSpecWorker.__new__(BaseSpecWorker)
                worker._target_worker = SimpleNamespace(model_runner=runner)
                with patch(
                    f"{_RUNNER_MODULE}.get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(sampling_backend="pytorch")
                    ),
                ):
                    result = worker.prewarm_sampling()
                self.loader.assert_called_once_with()
                self.tree_loader.assert_called_once_with(129280)
                self.assertIs(result, runner.sampling_prewarm_result)

    def test_unneeded_paths_do_not_load_flashinfer(self):
        for overrides in (
            {"device": "cpu"},
            {"device": "xpu"},
            {"device": "npu"},
            {"is_generation": False},
            {"is_draft_worker": True},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(
                    self._runner(**overrides).prewarm_sampling(),
                    SamplingPrewarmResult(),
                )
        self.loader.assert_not_called()
        self.tree_loader.assert_not_called()

    def test_hip_does_not_load_flashinfer_despite_cuda_device_name(self):
        with patch(f"{_RUNNER_MODULE}.is_cuda", return_value=False):
            self._runner(algorithm="EAGLE").prewarm_sampling()
        self.loader.assert_not_called()

    def test_regular_pytorch_sampler_does_not_load_flashinfer(self):
        with patch(
            f"{_RUNNER_MODULE}.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="pytorch")
            ),
        ):
            self._runner().prewarm_sampling()
        self.loader.assert_not_called()

    def test_build_and_load_errors_propagate_from_both_paths(self):
        for algorithm in ("NONE", "EAGLE"):
            for error in (
                RuntimeError("Ninja build failed"),
                OSError("cannot load so"),
            ):
                with self.subTest(algorithm=algorithm, error=error):
                    self.loader.side_effect = error
                    runner = self._runner(algorithm=algorithm)
                    if algorithm == "EAGLE":
                        worker = BaseSpecWorker.__new__(BaseSpecWorker)
                        worker._target_worker = SimpleNamespace(model_runner=runner)
                    else:
                        worker = runner
                    with self.assertRaises(type(error)) as raised:
                        worker.prewarm_sampling()
                    self.assertIs(raised.exception, error)

    def test_tree_build_failure_stops_eagle_startup(self):
        error = RuntimeError("tree sampling build failed")
        self.tree_loader.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self._runner(algorithm="EAGLE").prewarm_sampling()
        self.assertIs(raised.exception, error)
        self.loader.assert_called_once_with()

    def test_rejection_sampling_does_not_build_unused_tree_kernel(self):
        with patch(
            f"{_RUNNER_MODULE}.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=True),
        ):
            self._runner(algorithm="EAGLE").prewarm_sampling()
        self.loader.assert_called_once_with()
        self.tree_loader.assert_not_called()


class TestTreeSamplingPrewarm(CustomTestCase):
    def test_specialization_matches_request_vocab(self):
        for vocab_size, vec_size in ((129280, 4), (32770, 2), (32771, 1)):
            with self.subTest(vocab_size=vocab_size):
                # Exercise the real prewarm function, outside setUp's mock.
                with patch.object(tree_sampling, "_jit_sampling_module") as loader:
                    tree_sampling.prewarm_tree_speculative_sampling(vocab_size)
                loader.assert_called_once_with(vec_size, True)


class TestSamplingValidation(CustomTestCase):
    def test_aging_only_changes_private_sampling_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sampling = root / ".cache/flashinfer/0.6.18/103a/cached_ops/sampling"
            sampling.mkdir(parents=True)
            outputs = [sampling / "sampling.so", sampling / "sampling.cuda.o"]
            untouched = [sampling / "build.ninja", root / "other.so"]
            for path in outputs + untouched:
                path.write_text("test")
            before = {path: path.stat().st_mtime_ns for path in untouched}
            self.assertEqual(validation.age_sampling_artifacts(root), 2)
            for path in outputs:
                self.assertEqual(path.stat().st_mtime_ns, 1_000_000_000)
            for path in untouched:
                self.assertEqual(path.stat().st_mtime_ns, before[path])

    def test_cuda_subprocess_failure_is_not_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                validation.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "worker"),
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    validation.run_checks(output, 32, 10)
            env = run.call_args.kwargs["env"]
            for key, value in validation.cache_environment(output / "cache").items():
                self.assertEqual(env[key], value)
                self.assertTrue(Path(value).is_relative_to(output))
            self.assertFalse((output / "results.json").exists())

    def test_recompile_on_warm_start_is_a_failure(self):
        def worker(command, **kwargs):
            Path(command[-1]).write_text(json.dumps({"kernel_outputs": "passed"}))

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(validation.subprocess, "run", side_effect=worker),
                patch.object(
                    validation,
                    "build_artifacts",
                    side_effect=[{"sampling.so": (1, 2)}, {"sampling.so": (1, 3)}],
                ),
            ):
                with self.assertRaisesRegex(AssertionError, "Warm-cache"):
                    validation.run_checks(Path(directory), 32, 10)


if __name__ == "__main__":
    unittest.main()
