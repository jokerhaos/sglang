"""Unit tests for the NVDEC backend of sglang.srt.utils.video_decoder.

Everything here runs on a CPU-only machine: PyNvVideoCodec is faked with a
module-level monkeypatch, and the fake decoder hands out CPU torch tensors,
which exercises the same DLPack->numpy conversion the real DecodedFrame takes.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.utils import video_decoder as vd
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --- container probe (pure bytes -> codec routing) ---------------------------


@pytest.mark.parametrize(
    "blob,expected",
    [
        (b"\x00" * 64 + b"avc1" + b"\x00" * 64, "h264"),
        (b"\x00" * 64 + b"avc3" + b"\x00" * 64, "h264"),
        (b"\x00" * 64 + b"V_MPEG4/ISO/AVC" + b"\x00" * 64, "h264"),
        (b"\x00" * 64 + b"hvc1" + b"\x00" * 64, "hevc"),
        (b"\x00" * 64 + b"V_MPEGH/ISO/HEVC" + b"\x00" * 64, "hevc"),
        # hevc markers must not be shadowed by the h264 ones
        (b"\x00" * 64 + b"hev1" + b"\x00" * 64 + b"avc1", "hevc"),
        (b"\x00" * 64 + b"av01" + b"\x00" * 64, "av1"),
        (b"\x00" * 64 + b"vp09" + b"\x00" * 64, "vp9"),
        (b"", None),
        (b"\x00" * 128, None),
    ],
)
def test_probe_video_codec(blob, expected):
    assert vd.probe_video_codec(blob) == expected


# --- environment knobs -------------------------------------------------------


def test_disable_env(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_NVDEC", "1")
    vd._nvdec_available.cache_clear()
    try:
        assert vd._nvdec_available() is False
    finally:
        vd._nvdec_available.cache_clear()


def test_disable_env_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_DISABLE_NVDEC", raising=False)
    vd._nvdec_available.cache_clear()
    try:
        # Whatever the answer is on this machine, it must not depend on the
        # kill switch being absent-vs-empty.
        assert isinstance(vd._nvdec_available(), bool)
    finally:
        vd._nvdec_available.cache_clear()


@pytest.mark.parametrize("raw,expected", [("", 0), ("3", 3), ("not-a-number", 0)])
def test_gpu_id(monkeypatch, raw, expected):
    monkeypatch.setenv("SGLANG_NVDEC_GPU_ID", raw)
    assert vd._gpu_id() == expected


# --- NVDEC backend against a fake PyNvVideoCodec ---------------------------


class FakeSimpleDecoder:
    """Stands in for PyNvVideoCodec.SimpleDecoder on a CPU-only machine."""

    FPS = 25.0
    NUM_FRAMES = 8
    SHAPE = (240, 320)  # (H, W)

    def __init__(self, path, gpu_id=0, output_color_type=None, use_device_memory=False):
        self.path = path
        self.gpu_id = gpu_id
        self.output_color_type = output_color_type
        self.use_device_memory = use_device_memory
        self._frames = [
            torch.full((self.SHAPE[0], self.SHAPE[1], 3), i, dtype=torch.uint8)
            for i in range(self.NUM_FRAMES)
        ]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[int(idx)]

    def get_stream_metadata(self):
        return SimpleNamespace(
            average_fps=self.FPS,
            num_frames=self.NUM_FRAMES,
            duration=self.NUM_FRAMES / self.FPS,
        )


@pytest.fixture()
def fake_nvdec(monkeypatch):
    """Make the NVDEC path appear available and backed by a fake module."""
    fake_module = SimpleNamespace(
        SimpleDecoder=FakeSimpleDecoder,
        OutputColorType=SimpleNamespace(RGB="RGB"),
    )
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_module)
    # _nvdec_available is lru_cached over the real import; stub it so the test
    # does not depend on a real GPU stack.
    monkeypatch.setattr(vd, "_nvdec_available", lambda: True)
    monkeypatch.setattr(vd, "_is_cpu_engine", lambda: True)


def h264_bytes(marker=b"avc1"):
    return b"\x00" * 64 + marker + b"\x00" * 64


def test_nvdec_backend_selected_and_contract(fake_nvdec):
    wrapper = vd.VideoDecoderWrapper(h264_bytes())
    assert wrapper._backend == "nvdec"

    assert len(wrapper) == 8
    assert wrapper.avg_fps == 25.0
    assert wrapper.frame_shape == (240, 320)

    frames = wrapper.get_frames_at([0, 3, 7])
    assert frames.shape == (3, 240, 320, 3)
    assert frames.dtype == np.uint8
    # frame identity: torch.full(i) round-trips through DLPack as value i
    assert frames[0].min() == 0 and frames[1].min() == 3 and frames[2].min() == 7

    single = wrapper[5]
    assert single.shape == (240, 320, 3) and single.dtype == np.uint8

    tensor = wrapper.get_frames_as_tensor([1, 2])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 240, 320, 3) and tensor.dtype == torch.uint8

    # bytes source stays readable (audio extraction path) and the temp file
    # used by the decoder is cleaned up on close
    assert wrapper.source_bytes == wrapper._source_bytes
    tmp = wrapper._tmp_path
    assert tmp is not None and os.path.exists(tmp)
    wrapper.close()
    assert not os.path.exists(tmp)


def test_nvdec_skipped_for_royalty_free_codecs(fake_nvdec):
    # VP8/VP9/AV1 stay on the software chain even when NVDEC is available.
    assert vd.VideoDecoderWrapper._should_use_nvdec(h264_bytes(b"vp09")) is False
    assert vd.VideoDecoderWrapper._should_use_nvdec(h264_bytes(b"av01")) is False


def test_nvdec_skipped_when_probe_is_inconclusive(fake_nvdec):
    assert vd.VideoDecoderWrapper._should_use_nvdec(b"\x00" * 128) is False


def test_nvdec_open_failure_falls_back_to_software(fake_nvdec, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bad stream")

    boom_module = SimpleNamespace(
        SimpleDecoder=boom,
        OutputColorType=SimpleNamespace(RGB="RGB"),
    )
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", boom_module)
    monkeypatch.setattr(vd, "_BACKEND", "torchcodec")
    monkeypatch.setattr(vd, "VideoDecoder", lambda source, **kwargs: FakeSimpleDecoder(source))

    wrapper = vd.VideoDecoderWrapper(h264_bytes())
    assert wrapper._backend == "torchcodec"
    assert len(wrapper) == 8  # the software decoder took over


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
