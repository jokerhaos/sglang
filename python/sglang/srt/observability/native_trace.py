"""Optional W3C handoff to Mooncake bindings; tracing never changes IO outcomes."""

import logging
from contextlib import contextmanager
from functools import lru_cache

from sglang.srt.observability import trace as tracing

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warn_native_context_failure():
    logger.warning("Native trace context failed; continuing IO")


def capture_trace_context(context=None):
    carrier = {}
    if tracing.opentelemetry_initialized:
        tracing.TraceContextTextMapPropagator().inject(carrier, context=context)
    return carrier


def _clear_context(target):
    clear = getattr(target, "clear_trace_context", None)
    if clear is not None:
        try:
            clear()
        except Exception:
            pass


@contextmanager
def native_trace_scope(target, carrier):
    setter = getattr(target, "set_trace_context", None)
    previous = None
    if setter is not None and tracing.opentelemetry_initialized:
        try:
            previous = setter(
                carrier.get("traceparent", ""), carrier.get("tracestate", "")
            )
            if not (
                isinstance(previous, tuple)
                and len(previous) == 2
                and all(isinstance(value, str) for value in previous)
            ):
                previous = None
                _warn_native_context_failure()
                _clear_context(target)
        except Exception:
            _warn_native_context_failure()
            _clear_context(target)
    try:
        yield
    finally:
        if previous is not None:
            try:
                setter(*previous)
            except Exception:
                _warn_native_context_failure()
                _clear_context(target)
