"""Explicit, queue-safe tracing for HiCache storage operations."""

import logging
import threading
from contextlib import contextmanager
from functools import lru_cache

from sglang.srt.observability import trace as tracing

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warn_native_context_failure():
    logger.warning("HiCache native trace context failed; continuing cache IO")


def request_carrier(trace_ctx):
    carrier = {}
    parent = getattr(trace_ctx, "root_span_context", None)
    if tracing.opentelemetry_initialized and parent is not None:
        tracing.TraceContextTextMapPropagator().inject(carrier, context=parent)
    return carrier


class StorageTrace:
    """One operation, with an explicit parent and an idempotent terminal state.

    Shared writes are independent operations: a tree node has no single request
    owner. Request prefetches require a valid W3C parent, never ambient context.
    """

    def __init__(self, carrier=None, *, shared=False, requested_tokens=0):
        self.span = self.wait_span = self.context = None
        self._lock = threading.Lock()
        if (
            not tracing.opentelemetry_initialized
            or tracing.get_global_trace_level() <= 0
            or (
                tracing.global_trace_modules is not None
                and "hicache" not in tracing.global_trace_modules
            )
        ):
            return
        parent = tracing.context.Context()
        if not shared:
            parent = tracing.TraceContextTextMapPropagator().extract(carrier or {})
            if not tracing.trace.get_current_span(parent).get_span_context().is_valid:
                return
        self.span = tracing.tracer.start_span(
            "hicache.writeback" if shared else "hicache.prefetch",
            context=parent,
            attributes={
                "hicache.shared": shared,
                "hicache.requested_tokens": requested_tokens,
            },
        )
        self.context = tracing.trace.set_span_in_context(self.span, parent)
        if not shared:
            self.wait_span = tracing.tracer.start_span(
                "hicache.wait", context=self.context
            )

    def finish(self, outcome, completed_tokens=0):
        with self._lock:
            if self.span is None:
                return
            for span in (self.wait_span, self.span):
                if span is not None:
                    span.set_attribute("hicache.outcome", outcome)
                    span.set_attribute("hicache.completed_tokens", completed_tokens)
                    if outcome == "error":
                        span.set_status(tracing.Status(tracing.StatusCode.ERROR))
                    span.end()
            self.span = self.wait_span = None

    @contextmanager
    def stage(self, name, backend=None):
        if self.context is None:
            yield None
            return
        with tracing.tracer.start_as_current_span(
            "hicache." + name, context=self.context
        ) as span:
            store = getattr(backend, "store", None)
            setter = getattr(store, "set_trace_context", None)
            previous = None
            try:
                if setter is not None:
                    carrier = {}
                    tracing.TraceContextTextMapPropagator().inject(carrier)
                    try:
                        previous = setter(
                            carrier.get("traceparent", ""),
                            carrier.get("tracestate", ""),
                        )
                    except Exception:
                        _warn_native_context_failure()
                        clear = getattr(store, "clear_trace_context", None)
                        if clear is not None:
                            try:
                                clear()
                            except Exception:
                                pass
                yield span
            except Exception:
                self.finish("error")
                raise
            finally:
                if previous is not None:
                    try:
                        setter(*previous)
                    except Exception:
                        _warn_native_context_failure()
                        clear = getattr(store, "clear_trace_context", None)
                        if clear is not None:
                            try:
                                clear()
                            except Exception:
                                pass
