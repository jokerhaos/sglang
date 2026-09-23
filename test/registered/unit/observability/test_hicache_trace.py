"""CPU coverage of explicit HiCache request context and async operation lifetime."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.observability import trace as tracing
from sglang.srt.observability.hicache_trace import StorageTrace, request_carrier
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class NativeStore:
    def __init__(self):
        self.local = threading.local()

    def set_trace_context(self, traceparent, tracestate=""):
        previous = getattr(self.local, "carrier", ("", ""))
        self.local.carrier = (traceparent, tracestate)
        return previous


class TestHiCacheTrace(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        # Root writes follow configured sampling; sampled request parents win.
        self.provider = TracerProvider(sampler=ParentBased(ALWAYS_OFF))
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer("hicache-test")
        for name, value in (
            ("opentelemetry_initialized", True),
            ("tracer", self.tracer),
            ("global_trace_modules", ["hicache"]),
        ):
            patcher = patch.object(tracing, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(tracing, "get_global_trace_level", return_value=1)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.provider.shutdown)
        self.carrier = {
            "traceparent": "00-12345678901234567890123456789012-1234567890123456-01",
            "tracestate": "vendor=value",
        }

    def test_cross_thread_parent_native_restore_and_idempotent_finish(self):
        operation = StorageTrace(self.carrier, requested_tokens=32)
        backend = SimpleNamespace(store=NativeStore())

        def run():
            backend.store.set_trace_context("previous", "state")
            with operation.stage("query", backend) as span:
                carrier = backend.store.local.carrier
                self.assertIn("12345678901234567890123456789012", carrier[0])
                self.assertTrue(carrier[0].endswith("-01"))
                self.assertEqual(carrier[1], "vendor=value")
                self.assertEqual(
                    span.get_span_context().trace_id,
                    int(self.carrier["traceparent"].split("-")[1], 16),
                )
            self.assertEqual(backend.store.local.carrier, ("previous", "state"))
            with operation.stage("read", backend):
                pass

        with ThreadPoolExecutor(1) as executor:
            executor.submit(run).result()
        operation.finish("complete", 32)
        operation.finish("cancelled")
        spans = {s.name: s for s in self.exporter.get_finished_spans()}
        self.assertEqual(
            set(spans),
            {"hicache.prefetch", "hicache.wait", "hicache.query", "hicache.read"},
        )
        root = spans["hicache.prefetch"]
        self.assertEqual(root.parent.span_id, int("1234567890123456", 16))
        for name in ("hicache.wait", "hicache.query", "hicache.read"):
            self.assertEqual(spans[name].parent.span_id, root.context.span_id)
        self.assertEqual(root.attributes["hicache.completed_tokens"], 32)

    def test_unsampled_carrier_reaches_native_without_export(self):
        self.carrier["traceparent"] = self.carrier["traceparent"][:-2] + "00"
        operation = StorageTrace(self.carrier)
        backend = SimpleNamespace(store=NativeStore())
        with operation.stage("read", backend):
            self.assertTrue(backend.store.local.carrier[0].endswith("-00"))
        operation.finish("complete")
        self.assertEqual(self.exporter.get_finished_spans(), ())

    def test_disabled_filtered_invalid_do_not_generate_ids(self):
        for overrides in (
            {"opentelemetry_initialized": False},
            {"global_trace_modules": ["request"]},
        ):
            with (
                patch.multiple(tracing, **overrides),
                patch.object(self.tracer, "start_span") as start,
            ):
                operation = StorageTrace(self.carrier)
                with operation.stage("read"):
                    pass
                operation.finish("complete")
                start.assert_not_called()
        with patch.object(self.tracer, "start_span") as start:
            StorageTrace({"traceparent": "invalid"}).finish("fallback")
            start.assert_not_called()

    def test_exception_and_cancellation(self):
        backend = SimpleNamespace(store=NativeStore())
        operation = StorageTrace(self.carrier)
        with self.assertRaisesRegex(RuntimeError, "IO failed"):
            with operation.stage("read", backend):
                raise RuntimeError("IO failed")
        self.assertEqual(backend.store.local.carrier, ("", ""))
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        self.assertTrue(
            all(s.status.status_code == trace.StatusCode.ERROR for s in spans)
        )

    def test_terminal_outcomes_preserve_status_and_first_result(self):
        for outcome in (
            "complete",
            "miss",
            "below_threshold",
            "allocation_failed",
            "cancelled",
            "timeout",
            "early_stop",
            "fallback",
            "partial",
            "skipped",
            "error",
        ):
            with self.subTest(outcome=outcome):
                self.exporter.clear()
                operation = StorageTrace(self.carrier)
                operation.finish(outcome, 8)
                spans = self.exporter.get_finished_spans()
                self.assertEqual(len(spans), 2)
                self.assertEqual(
                    {span.name for span in spans}, {"hicache.prefetch", "hicache.wait"}
                )
                end_times = [span.end_time for span in spans]
                operation.finish("complete", 16)
                operation.finish("error", 0)
                final_spans = self.exporter.get_finished_spans()
                self.assertEqual(len(final_spans), 2)
                self.assertEqual([span.end_time for span in final_spans], end_times)
                for span in final_spans:
                    self.assertEqual(span.attributes["hicache.outcome"], outcome)
                    self.assertEqual(span.attributes["hicache.completed_tokens"], 8)
                    self.assertEqual(
                        span.status.status_code,
                        trace.StatusCode.ERROR
                        if outcome == "error"
                        else trace.StatusCode.UNSET,
                    )

    def test_shared_write_does_not_inherit_ambient_request(self):
        parent = tracing.TraceContextTextMapPropagator().extract(self.carrier)
        with trace.use_span(trace.get_current_span(parent)):
            operation = StorageTrace(shared=True)
            self.assertNotEqual(
                trace.get_current_span(operation.context).get_span_context().trace_id,
                trace.get_current_span(parent).get_span_context().trace_id,
            )
            operation.finish("complete")
        self.assertEqual(self.exporter.get_finished_spans(), ())

    def test_request_carrier_uses_w3c_and_preserves_state(self):
        context = tracing.TraceContextTextMapPropagator().extract(self.carrier)
        self.assertEqual(
            request_carrier(SimpleNamespace(root_span_context=context)), self.carrier
        )

    def test_real_controller_query_read_write_workers(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.storage_stop_event = threading.Event()
        controller.prefetch_queue = Queue()
        controller.prefetch_hit_queue = Queue()
        controller.prefetch_buffer = Queue()
        controller.prefetch_sync_queue = Queue()
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.backup_skip = False
        controller.prefetch_hits_sync_groups = []
        controller._all_reduce = lambda *args: None
        store = NativeStore()
        observed = []

        def exists(keys, extra_info):
            observed.append(store.local.carrier)
            controller.storage_stop_event.set()
            return len(keys)

        controller.storage_backend = SimpleNamespace(store=store, batch_exists=exists)
        operation = controller.prefetch("req", [1, 2, 3, 4], trace_carrier=self.carrier)
        with patch(
            "sglang.srt.managers.cache_controller.get_storage_hash_str",
            return_value=["a", "b"],
        ):
            controller.prefetch_thread_func()
        self.assertIs(controller.prefetch_hit_queue.get_nowait(), operation)
        self.assertEqual(operation.storage_hit_count, 4)
        operation.host_indices = [0, 1, 2, 3]

        def get_pages(*args):
            observed.append(store.local.carrier)
            controller.storage_stop_event.set()
            return 2

        controller.page_get_func = get_pages
        controller.storage_stop_event.clear()
        controller.prefetch_buffer.put(operation)
        controller.prefetch_io_aux_func()
        ack = controller.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.completed_tokens, 4)
        self.assertTrue(controller.prefetch_sync_queue.get_nowait().completed_req)
        operation.storage_trace.finish("complete", ack.completed_tokens)
        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all("12345678901234567890123456789012" in item[0] for item in observed)
        )

        def set_pages(*args):
            controller.storage_stop_event.set()
            return False

        controller.page_set_func = set_pages
        controller.storage_stop_event.clear()
        controller.write_storage([0, 1], [1, 2], hash_value=["a"])
        # Use a sampled parent for inspecting write failure closure, independently
        # of the shared-write root sampler behavior covered above.
        write = controller.backup_queue.get_nowait()
        write.storage_trace.finish("cancelled")
        write.storage_trace = StorageTrace(self.carrier)
        controller.backup_queue.put(write)
        controller.backup_thread_func()
        self.assertIs(controller.ack_backup_queue.get_nowait(), write)
        failures = [
            s
            for s in self.exporter.get_finished_spans()
            if s.attributes.get("hicache.outcome") == "error"
        ]
        self.assertEqual(len(failures), 2)

    def test_real_cache_completed_prefetch_is_not_timeout(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.prefetch_queue = Queue()
        operation = controller.prefetch("req", [1, 2], trace_carrier=self.carrier)
        operation.host_indices = [0, 1]
        operation.completed_tokens = operation.storage_hit_count = 2
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.pp_rank = 0
        cache.prefetch_stop_policy = "timeout"
        cache.is_prefetch_timeout = lambda op: True
        cache._all_reduce = lambda *args: None
        cache.cache_controller = controller
        cache.ongoing_prefetch = {"req": (None, None, operation)}
        cache._handle_prefetch_result = lambda op: op.storage_trace.finish(
            "complete", 2
        )
        self.assertTrue(cache.check_prefetch_progress(SimpleNamespace(rid="req")))
        self.assertTrue(
            all(
                s.attributes["hicache.outcome"] == "complete"
                for s in self.exporter.get_finished_spans()
            )
        )

    def test_native_failure_is_observability_only(self):
        for fail_on in (1, 2):
            calls = []

            def setter(*args):
                calls.append(args)
                if len(calls) == fail_on:
                    raise RuntimeError("native tracing unavailable")
                return ("", "")

            operation = StorageTrace(self.carrier)
            with operation.stage(
                "read", SimpleNamespace(store=SimpleNamespace(set_trace_context=setter))
            ):
                pass
            operation.finish("complete")
        self.assertTrue(
            all(
                s.status.status_code != trace.StatusCode.ERROR
                for s in self.exporter.get_finished_spans()
            )
        )


if __name__ == "__main__":
    unittest.main()
