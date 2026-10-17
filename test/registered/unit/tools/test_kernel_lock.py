import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = load_module(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKernelLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_module("kernel_lock", REPO_ROOT / "docker/kernel_lock.py")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        for name in ("pyproject.toml", "kernels.lock", "kernels.lock.inputs.json"):
            shutil.copyfile(REPO_ROOT / "python" / name, self.project / name)

    def test_checked_in_lock_is_valid_without_network(self):
        self.helper.check_lock(self.project)

    def test_changed_kernel_dependency_requires_regeneration(self):
        (self.project / "pyproject.toml").write_text(
            '[tool.kernels.dependencies]\n"kernels-community/sgl-flash-attn3" = 2\n'
        )
        with self.assertRaisesRegex(ValueError, "Kernel dependencies changed"):
            self.helper.check_lock(self.project)

    def test_source_only_change_keeps_lock_valid(self):
        with (self.project / "pyproject.toml").open("a") as source:
            source.write("\n[tool.kernel_lock_test]\nsource_only = true\n")
        self.helper.check_lock(self.project)

    def test_modified_lock_is_rejected(self):
        lock = self.project / "kernels.lock"
        lock.write_text(lock.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.helper.check_lock(self.project)

    def test_missing_lock_is_rejected(self):
        (self.project / "kernels.lock").unlink()
        with self.assertRaises(FileNotFoundError):
            self.helper.check_lock(self.project)

    def test_native_kernel_lock_format_is_preserved(self):
        from kernels.lockfile import KernelLock

        entries = json.loads((self.project / "kernels.lock").read_text())
        self.assertTrue(entries)
        for entry in entries:
            lock = KernelLock.from_json(entry)
            self.assertEqual(len(lock.sha), 40)
            self.assertTrue(lock.variants)


if __name__ == "__main__":
    unittest.main()
