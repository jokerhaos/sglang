"""Generate or validate the kernel lock used by local-source image builds."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import tomllib
from pathlib import Path


def dependencies(project: Path) -> dict:
    with (project / "pyproject.toml").open("rb") as source:
        return (
            tomllib.load(source)
            .get("tool", {})
            .get("kernels", {})
            .get("dependencies", {})
        )


def check_lock(project: Path) -> None:
    inputs = json.loads((project / "kernels.lock.inputs.json").read_text())
    if inputs["dependencies"] != dependencies(project):
        raise ValueError("Kernel dependencies changed; regenerate kernels.lock")
    digest = hashlib.sha256((project / "kernels.lock").read_bytes()).hexdigest()
    if digest != inputs["sha256"]:
        raise ValueError("kernels.lock does not match its generated input record")


def update_lock(project: Path) -> None:
    subprocess.run(["kernels", "lock", str(project)], check=True)
    lock = project / "kernels.lock"
    lock.write_text(json.dumps(json.loads(lock.read_text()), indent=2) + "\n")
    inputs = {
        "dependencies": dependencies(project),
        "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "generator": f"kernels {importlib.metadata.version('kernels')}",
    }
    (project / "kernels.lock.inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    if args.update:
        update_lock(args.project)
    check_lock(args.project)
