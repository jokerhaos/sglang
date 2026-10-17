import hashlib
import importlib.util
import io
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestInstallBuildctl(unittest.TestCase):
    def setUp(self):
        if sys.platform != "linux" or any(
            shutil.which(tool) is None
            for tool in ("bash", "curl", "flock", "sha256sum", "tar")
        ):
            self.skipTest("Requires the Linux runner's build tools")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.binary = self.root / "bin/buildctl"
        data = b'#!/bin/sh\nprintf "buildctl fixture v0.33.0\\n"\n'
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            entry = tarfile.TarInfo("bin/buildctl")
            entry.size, entry.mode = len(data), 0o755
            archive.addfile(entry, io.BytesIO(data))
        self.payload = buffer.getvalue()
        self.sha256 = hashlib.sha256(self.payload).hexdigest()
        self.requests = []
        self.interrupt_first = False
        self.ignore_range = False
        self.corrupt_download = False
        self.cut = len(self.payload) // 2
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                byte_range = self.headers.get("Range")
                fixture.requests.append(byte_range)
                body = (
                    b"corrupt archive" if fixture.corrupt_download else fixture.payload
                )
                offset = (
                    int(byte_range.split("=")[1].split("-")[0]) if byte_range else 0
                )
                if offset >= len(body):
                    self.send_response(416)
                    self.end_headers()
                    return
                resumed = byte_range is not None and not fixture.ignore_range
                if resumed:
                    body = body[offset:]
                self.send_response(206 if resumed else 200)
                if resumed:
                    self.send_header(
                        "Content-Range",
                        f"bytes {offset}-{len(fixture.payload) - 1}/{len(fixture.payload)}",
                    )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                interrupted = fixture.interrupt_first and len(fixture.requests) == 1
                try:
                    self.wfile.write(body[: fixture.cut] if interrupted else body)
                    self.wfile.flush()
                    if interrupted:
                        self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            self.server.shutdown()
            self.server.server_close()
            thread.join()

        self.addCleanup(stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/buildkit.tar.gz"
        # Avoid production retry delays while exercising the actual curl/tar
        # commands and HTTP range handling against the local fixture server.
        commands = self.root / "commands"
        commands.mkdir()
        sleep = commands / "sleep"
        sleep.write_text("#!/bin/sh\nexit 0\n")
        sleep.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"}

    def install(self, binary=None):
        return subprocess.run(
            self.command(binary),
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def command(self, binary=None):
        return [
            "bash",
            str(REPO_ROOT / "scripts/ci/utils/install_buildctl.sh"),
            self.url,
            self.sha256,
            str(self.cache),
            str(binary or self.binary),
        ]

    def assert_installed(self, result, binary=None):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("buildctl fixture v0.33.0", result.stdout)
        self.assertTrue(os.access(binary or self.binary, os.X_OK))

    def test_fresh_download_is_verified_and_installed(self):
        self.assert_installed(self.install())
        self.assertEqual(self.requests, [None])

    def test_new_runner_reuses_shared_archive_without_network(self):
        self.assert_installed(self.install())
        other_binary = self.root / "other-runner/buildctl"
        self.assert_installed(self.install(other_binary), other_binary)
        self.assertEqual(self.requests, [None])

    def test_interrupted_download_resumes_at_saved_offset(self):
        self.interrupt_first = True
        self.assert_installed(self.install())
        self.assertEqual(self.requests, [None, f"bytes={self.cut}-"])

    def test_server_without_range_support_restarts_cleanly(self):
        self.interrupt_first, self.ignore_range = True, True
        self.assert_installed(self.install())
        self.assertEqual(self.requests, [None, f"bytes={self.cut}-", None])

    def test_corrupt_cached_archive_is_replaced(self):
        self.cache.mkdir()
        (self.cache / "buildkit.tar.gz").write_bytes(b"corrupt cache")
        self.assert_installed(self.install())
        self.assertEqual(self.requests, [None])

    def test_invalid_partial_offset_restarts_cleanly(self):
        self.cache.mkdir()
        (self.cache / "buildkit.tar.gz.part").write_bytes(
            b"x" * (len(self.payload) + 1)
        )
        self.assert_installed(self.install())
        self.assertEqual(self.requests, [f"bytes={len(self.payload) + 1}-", None])

    def test_bad_download_never_publishes_binary_or_archive(self):
        self.corrupt_download = True
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse(self.binary.exists())
        self.assertFalse((self.cache / "buildkit.tar.gz").exists())
        self.assertEqual(len(self.requests), 8)

    def test_concurrent_installers_share_one_download(self):
        other_binary = self.root / "other-runner/buildctl"
        processes = [
            subprocess.Popen(
                self.command(binary),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for binary in (self.binary, other_binary)
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("buildctl fixture v0.33.0", stdout)
        self.assertTrue(self.binary.exists())
        self.assertTrue(other_binary.exists())
        self.assertEqual(self.requests, [None])


if __name__ == "__main__":
    unittest.main()
