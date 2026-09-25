"""Unified video decoder: NVDEC (PyNvVideoCodec) or torchcodec, decord as fallback.

Backend priority per source:
  1. NVDEC -- when PyNvVideoCodec is importable AND the container probed from
     the source bytes is H.264/H.265. Decode runs on the GPU's dedicated NVDEC
     engines (no SM cost beyond a small YUV->RGB conversion), which is what the
     GPU serving images rely on: they ship no software H.264/H.265 decoder at
     all (codec-compliant images purge torchcodec/decord/av/opencv), so NVDEC
     is the only path there rather than merely the fastest.
  2. torchcodec (CPU, or CUDA via set_cuda_backend).
  3. decord.

NVDEC is selected independently of the 'device' argument because it changes
where decoding happens, not where the caller wants the frames: every backend
returns NHWC uint8 frames (numpy on CPU; tensor on CUDA only for torchcodec's
CUDA path). Set SGLANG_DISABLE_NVDEC=1 to force the software chain, and
SGLANG_NVDEC_GPU_ID to pin the decode GPU (default 0).

The codec probe is a routing hint over container fourcc markers, not a demux:
unknown containers simply stay on the software chain. A source that probes as
H.264/H.265 but fails to open on NVDEC (corrupt stream, unsupported profile)
falls back to the software chain -- except in images without any software
decoder, where the original error is re-raised as-is.
"""

import functools
import importlib.util
import logging
import os
import tempfile
import warnings

import numpy as np

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    _BACKEND = "torchcodec"
except (ImportError, RuntimeError):
    _BACKEND = "decord"


# ---------------------------------------------------------------------------
# NVDEC (PyNvVideoCodec) backend
# ---------------------------------------------------------------------------

DISABLE_ENV = "SGLANG_DISABLE_NVDEC"
GPU_ID_ENV = "SGLANG_NVDEC_GPU_ID"

# Codecs routed to NVDEC. Royalty-free codecs (VP8/VP9/AV1) stay on the
# software chain; NVDEC supports them too but there is nothing to gain when a
# compliant decode already exists, and keeping the set small makes the routing
# predictable.
HW_ROUTED_CODECS = frozenset({"h264", "hevc"})

# Used only when the decoder cannot report a source frame rate. Consumers
# divide by fps to build frame timestamps, so it must stay non-zero; 30 fps
# keeps timestamps plausible rather than failing the request.
FALLBACK_FPS = 30.0

# Container byte markers -> codec id. A video file carries exactly one video
# codec, so the first match wins. Covers ISO-BMFF (mp4/mov) sample-entry
# fourccs and Matroska/WebM CodecID strings. HEVC/AV1 checked before H.264 so
# a more specific marker is not shadowed.
_CODEC_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"hev1", "hevc"),
    (b"hvc1", "hevc"),
    (b"V_MPEGH/ISO/HEVC", "hevc"),
    (b"av01", "av1"),
    (b"V_AV1", "av1"),
    (b"avc1", "h264"),
    (b"avc3", "h264"),
    (b"V_MPEG4/ISO/AVC", "h264"),
    (b"vp09", "vp9"),
    (b"V_VP9", "vp9"),
    (b"vp08", "vp8"),
    (b"V_VP8", "vp8"),
)


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def probe_video_codec(data: bytes) -> str | None:
    """Best-effort codec identification from container bytes.

    Returns a codec id ("h264"/"hevc"/"vp9"/"vp8"/"av1") or None when it can't
    tell. Scans the whole buffer because an mp4 moov (which holds the codec
    fourcc) may sit at the end of a non-faststart file; multimodal clips are
    small and already in memory.
    """
    if not data:
        return None
    for marker, codec in _CODEC_MARKERS:
        if marker in data:
            return codec
    return None


def _gpu_id() -> int:
    raw = os.environ.get(GPU_ID_ENV, "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using GPU 0", GPU_ID_ENV, raw)
        return 0


@functools.lru_cache(maxsize=1)
def _nvdec_available() -> bool:
    """True if NVDEC decode can run in this process.

    Cached: the import is the reliable signal, and PyNvVideoCodec is only
    installed in GPU images. SGLANG_DISABLE_NVDEC forces the software path.

    A wheel being installed but failing to import is almost always a container
    that did not get the driver's video libraries (libnvcuvid) mounted, which
    silently costs H.264/H.265 decode on images with no software fallback --
    that case is warned loudly on purpose.
    """
    if _env_bool(DISABLE_ENV):
        return False
    try:
        with warnings.catch_warnings():
            # A third-party deprecation must not decide whether the capability
            # exists; under pytest's filterwarnings=error a warning IS raised,
            # the import "fails", and this lru_cache then pins False for the
            # whole process.
            warnings.simplefilter("ignore")
            import PyNvVideoCodec  # noqa: F401
    except Exception as exc:
        if importlib.util.find_spec("PyNvVideoCodec") is None:
            logger.debug(
                "PyNvVideoCodec is not installed; NVDEC decode disabled (%s)", exc
            )
        else:
            logger.warning(
                "PyNvVideoCodec is installed but failed to import (%s); NVDEC "
                "hardware decode is DISABLED. This usually means the container "
                "was not given the 'video' driver capability, so libnvcuvid is "
                "unavailable -- set NVIDIA_DRIVER_CAPABILITIES to include "
                "'video'. On images without a software video decoder, H.264/"
                "H.265 video input will fail until this is fixed.",
                exc,
            )
        return False
    return True


def _frame_to_rgb_hwc(frame) -> np.ndarray:
    """Copy a decoded RGB frame to a host (H, W, 3) uint8 array.

    A 2.x DecodedFrame (output_color_type=RGB) supports the DLPack protocol, so
    torch wraps it zero-copy and .cpu() copies to host; the as_tensor path
    covers objects that only expose the CUDA array interface.
    """
    import torch

    try:
        tensor = torch.from_dlpack(frame)
    except Exception:  # noqa: BLE001 - fall back to the CUDA-array-interface path
        tensor = torch.as_tensor(frame, device="cuda")
    arr = tensor.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _nvdec_source_fps(decoder) -> float:
    """Source frame rate from the decoder. Never returns 0.

    Consumers divide by fps to build frame timestamps, so a zero raises
    ZeroDivisionError and kills the request. Probes the metadata API across
    PyNvVideoCodec versions, then derives it from the frame span, then falls
    back so an API change degrades accuracy instead of failing requests.
    """
    meta = getattr(decoder, "get_stream_metadata", None)
    if callable(meta):
        try:
            info = meta()
            val = getattr(info, "average_fps", None)
            if val and float(val) > 0:
                return float(val)
            frames = float(getattr(info, "num_frames", 0) or 0)
            duration = float(getattr(info, "duration", 0) or 0)
            if frames > 0 and duration > 0:
                return frames / duration
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            pass

    for attr in ("get_fps", "fps", "GetFPS"):
        val = getattr(decoder, attr, None)
        try:
            val = val() if callable(val) else val
            if val and float(val) > 0:
                return float(val)
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            continue

    logger.warning(
        "NVDEC could not determine source fps; falling back to %.1f. Frame "
        "timestamps handed to the model will be approximate.",
        FALLBACK_FPS,
    )
    return FALLBACK_FPS


def _source_bytes_for_probe(source) -> bytes | None:
    """Read the source bytes for the codec probe. Returns None when unreadable."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, str) and os.path.isfile(source):
        try:
            with open(source, "rb") as f:
                return f.read()
        except OSError as exc:
            logger.debug("could not read %s for codec probe: %s", source, exc)
    return None


def _materialize_path(source) -> tuple[str, str | None]:
    """Return (path, temp_path) for a str-or-bytes source.

    Users of this module accept either a file path or raw bytes; NVDEC and
    decord both need a path on disk, so bytes go through a temp file whose
    lifetime the caller owns.
    """
    if isinstance(source, str):
        return source, None
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        os.write(fd, source)
    finally:
        os.close(fd)
    return tmp_path, tmp_path


_cuda_backend_enabled: bool | None = None


def _try_cuda_backend() -> bool:
    """Try to enable torchcodec CUDA backend. Caches result after first call."""
    global _cuda_backend_enabled
    if _cuda_backend_enabled is not None:
        return _cuda_backend_enabled
    try:
        from torchcodec.decoders import set_cuda_backend

        set_cuda_backend("beta")
        _cuda_backend_enabled = True
    except Exception:
        _cuda_backend_enabled = False
    return _cuda_backend_enabled


def _is_cpu_engine() -> bool:
    # Lazy import to avoid circular dependency issues and unnecessary imports on module load
    from sglang.srt.utils.common import is_cpu

    return is_cpu()


class VideoDecoderWrapper:
    """Unified video decoder: NVDEC (H.264/H.265) / torchcodec / decord.

    All frames are returned in NHWC uint8 numpy format for consistency (the
    torchcodec CUDA path returns CUDA tensors instead).

    num_decode_threads only applies to the torchcodec backend: NVDEC decodes
    fast enough on its dedicated hardware to not need parallel decoders.
    """

    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):
        """source: file path (str) or video bytes.
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec
            or NVDEC; NVDEC engages on codec + availability regardless of this
            argument (see the module docstring).
        num_decode_threads: number of parallel decoder instances for frame
            extraction (torchcodec only). 0 = auto (capped at 16),
            1 = single decoder. Set > 1 to split frame indices across
            multiple decoders in parallel threads.
        """
        self._source = source
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        self._backend = _BACKEND

        if self._should_use_nvdec(source):
            try:
                self._init_nvdec(source)
                return
            except Exception as exc:  # noqa: BLE001 - fall back to software chain
                # An H.264/H.265 stream that NVDEC cannot open (corrupt
                # container, unsupported profile) is exactly what the software
                # chain exists for. Losing the decoder then re-raises below.
                logger.warning(
                    "NVDEC open failed (%s); falling back to %s", exc, _BACKEND
                )
                self._cleanup_temp_file()

        if _BACKEND == "torchcodec":
            kwargs = {"dimension_order": "NHWC"}
            if device == "cuda" and _try_cuda_backend():
                kwargs["device"] = "cuda"
            self._tc_kwargs = kwargs
            try:
                self._decoder = VideoDecoder(source, **kwargs)
            except RuntimeError:
                if "device" in kwargs:
                    logger.warning("CUDA video decoding failed, falling back to CPU.")
                    kwargs.pop("device")
                    self._tc_kwargs = kwargs
                    self._decoder = VideoDecoder(source, **kwargs)
                else:
                    raise
        else:
            from decord import VideoReader, cpu

            path, tmp_path = _materialize_path(source)
            self._tmp_path = tmp_path
            self._decoder = VideoReader(path, ctx=cpu(0))

    # -- backend selection ------------------------------------------------

    @staticmethod
    def _should_use_nvdec(source) -> bool:
        """NVDEC only when the package imports, the codec routes there, and the
        source is probeable. The probe is a routing hint: containers without a
        known marker stay on the software chain."""
        if not _nvdec_available():
            return False
        data = _source_bytes_for_probe(source)
        if data is None:
            return False
        return probe_video_codec(data) in HW_ROUTED_CODECS

    def _init_nvdec(self, source):
        import PyNvVideoCodec as nvc

        path, tmp_path = _materialize_path(source)
        self._tmp_path = tmp_path
        # use_device_memory=False keeps frames host-visible immediately, which
        # the numpy return contract needs; the copy is the DLPack one.
        self._decoder = nvc.SimpleDecoder(
            path,
            gpu_id=_gpu_id(),
            output_color_type=nvc.OutputColorType.RGB,
            use_device_memory=False,
        )
        if len(self._decoder) <= 0:
            raise RuntimeError("NVDEC decoder reports zero frames")
        self._backend = "nvdec"

    def _cleanup_temp_file(self):
        if self._tmp_path is not None:
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            self._tmp_path = None

    # -- unified surface ---------------------------------------------------

    def __len__(self):
        return len(self._decoder)

    def __getitem__(self, idx):
        """Return one NHWC uint8 frame (numpy on CPU, tensor on CUDA)."""
        if self._backend == "nvdec":
            return _frame_to_rgb_hwc(self._decoder[int(idx)])
        elif self._backend == "torchcodec":
            frame = self._decoder[idx]
            data = frame.data if hasattr(frame, "data") else frame
            return data if data.is_cuda else data.numpy()
        else:
            frame = self._decoder[idx]
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)

    @property
    def avg_fps(self) -> float:
        if self._backend == "nvdec":
            return _nvdec_source_fps(self._decoder)
        elif self._backend == "torchcodec":
            return self._decoder.metadata.average_fps
        else:
            return self._decoder.get_avg_fps()

    @property
    def frame_shape(self) -> tuple[int, int]:
        if self._backend == "torchcodec":
            metadata = self._decoder.metadata
            height = getattr(metadata, "height", None)
            width = getattr(metadata, "width", None)
            if height and width:
                return int(height), int(width)
        shape = self[0].shape
        return int(shape[-3]), int(shape[-2])

    def get_frames_at(self, indices: list):
        """Return NHWC uint8 frames (numpy on CPU, tensor on CUDA)."""
        if self._backend == "nvdec":
            return np.stack([self[int(i)] for i in indices])
        elif self._backend == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            return batch.data if batch.data.is_cuda else batch.data.numpy()
        else:
            return self._decoder.get_batch(indices).asnumpy()

    def get_frames_as_tensor(self, indices: list):
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory)."""
        import torch

        if self._backend == "nvdec":
            output = torch.from_numpy(self.get_frames_at(indices))
            return output if _is_cpu_engine() else output.pin_memory()

        if (
            self._backend == "torchcodec"
            and self._num_decode_threads != 1
            and len(indices) > 1
        ):
            num_threads = self._num_decode_threads
            if num_threads <= 0:
                num_threads = min(os.cpu_count() or 8, 16)
            num_threads = min(num_threads, len(indices))
            if num_threads > 1:
                return self._parallel_decode(indices, num_threads)

        if self._backend == "torchcodec":
            batch = self._decoder.get_frames_at(indices)
            if _is_cpu_engine():
                return batch.data
            return batch.data if batch.data.is_cuda else batch.data.pin_memory()
        else:
            arr = self._decoder.get_batch(indices).asnumpy()
            output = torch.from_numpy(arr)
            return output if _is_cpu_engine() else output.pin_memory()

    def _parallel_decode(self, indices, num_threads):
        """Decode frames using multiple VideoDecoder instances in parallel threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]
        source = self._source
        kwargs = self._tc_kwargs
        cuda_device = None
        if kwargs.get("device") == "cuda":
            cuda_device = torch.cuda.current_device()

        def _decode_chunk(chunk):
            # CUDA's current device is thread-local.  Without this, decoder
            # workers created by TP rank > 0 silently default to GPU 0.
            if cuda_device is not None:
                torch.cuda.set_device(cuda_device)
            d = VideoDecoder(source, **kwargs)
            return d.get_frames_at(chunk).data

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_idx = {
                executor.submit(_decode_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            results = [None] * len(chunks)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        output = torch.cat(results, dim=0)
        if _is_cpu_engine():
            return output
        return output if output.is_cuda else output.pin_memory()

    @property
    def source_bytes(self) -> bytes | None:
        """Return raw video bytes if available (needed for audio extraction)."""
        if self._source_bytes is not None:
            return self._source_bytes
        path = self._tmp_path or self._source_path
        if path is not None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def close(self):
        self._decoder = None
        self._source = None
        self._source_bytes = None
        self._source_path = None
        if getattr(self, "_tmp_path", None) is not None:
            self._cleanup_temp_file()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
