"""Validate installed native tracing without requiring a GPU driver or collector."""

import ctypes
import importlib.metadata
import re
import subprocess
from pathlib import Path


def main():
    for name in (
        "mooncake-transfer-engine-cuda13",
        "mooncake-transfer-engine-non-cuda",
        "mooncake-transfer-engine",
    ):
        try:
            dist = importlib.metadata.distribution(name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        raise RuntimeError("Mooncake wheel is not installed")
    root = Path(dist.locate_file("mooncake"))
    libraries = set()
    for module in ("engine.so", "store.so"):
        # ldd resolves dependencies without importing these CUDA-linked modules.
        # A missing host libcuda is expected in a GPU-less Docker build.
        result = subprocess.run(
            ["ldd", str(root / module)], check=True, capture_output=True, text=True
        )
        matches = re.findall(
            r"^\s*libmooncake_native_tracing_impl\S*\s+=>\s+(\S+)",
            result.stdout,
            re.MULTILINE,
        )
        if len(matches) != 1 or not Path(matches[0]).is_file():
            raise RuntimeError(f"{module} must resolve one native tracing library")
        libraries.add(Path(matches[0]).resolve())
    if len(libraries) != 1:
        raise RuntimeError("engine/store resolve different native tracing libraries")
    library = libraries.pop()
    if library not in {Path(dist.locate_file(f)).resolve() for f in dist.files or []}:
        raise RuntimeError("Native tracing library must be bundled in the wheel")
    if b"MC_NATIVE_TRACING_ENABLED" not in library.read_bytes():
        raise RuntimeError("Native library lacks tracing initialization")
    # This DSO has no CUDA dependency; resolve all of its runtime dependencies.
    ctypes.CDLL(str(library))
    print("Mooncake native tracing: shared dependency and runtime loading verified")


if __name__ == "__main__":
    main()
