"""CPU tests of the real P-D worker and layer-executor native handoffs."""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.observability import trace as tracing
from sglang.srt.observability.native_trace import native_trace_scope
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class StopWorker(BaseException):
    pass


class Engine:
    def __init__(self):
        self.local = threading.local()
        self.calls = []

    def set_trace_context(self, parent, state=""):
        previous = getattr(self.local, "carrier", ("", ""))
        self.local.carrier = (parent, state)
        return previous

    def batch_transfer_sync(self, session, sources, targets, lengths):
        self.calls.append((self.local.carrier, sources, targets, lengths))
        return 0


class TestMooncakeNativeTrace(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        patcher = patch.multiple(
            tracing,
            opentelemetry_initialized=True,
            tracer=self.provider.get_tracer("native-test"),
            global_trace_modules=["mooncake"],
            threads_info={},
            _trace_context_propagator=tracing.TraceContextTextMapPropagator(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.provider.shutdown)

    def manager(self):
        manager = object.__new__(MooncakeKVManager)
        manager.engine = Engine()
        manager.enable_trace = True
        manager.attn_tp_rank = manager.attn_dp_rank = 0
        manager.attn_tp_size = manager.pp_size = 1
        manager.is_mla_backend = manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = True
        manager.custom_mem_pool_type = "TEST"
        manager.enable_staging = False
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {1: object()}
        manager.check_status = lambda room: object()
        manager.transfer_infos = {}
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_item_lens=[8],
            page_size=1,
            total_kv_head_num=1,
            kv_data_ptrs=[100],
            kv_layer_ids=[0],
            num_draft_entries=0,
        )
        return manager

    def test_worker_slice_through_all_layer_executors(self):
        for flags in ("01", "00"):
            for path in ("generic", "tp", "dcp", "direct"):
                with (
                    self.subTest(flags=flags, path=path),
                    ThreadPoolExecutor(1) as pool,
                ):
                    manager = self.manager()
                    ctx = tracing.TraceReqContext(
                        "request",
                        module_name="mooncake",
                        trace_level=3,
                        external_trace_header={
                            "traceparent": f"00-12345678901234567890123456789012-1234567890123456-{flags}",
                            "tracestate": "vendor=value",
                        },
                    )
                    ctx.trace_req_start()
                    chunk = SimpleNamespace(
                        trace_ctx=ctx, room=1, staging_counted=False
                    )
                    indices = np.array([0], dtype=np.int32)

                    def transfer(
                        manager=manager,
                        ctx=ctx,
                        path=path,
                        flags=flags,
                        pool=pool,
                        indices=indices,
                    ):
                        if path == "generic":
                            manager._send_kvcache_generic(
                                "session",
                                [100],
                                [200],
                                [8],
                                indices,
                                indices,
                                pool,
                                src_layer_ids=[0],
                                dst_layer_ids=[0],
                            )
                        elif path == "tp":
                            manager.send_kvcache_slice(
                                "session",
                                indices,
                                [200],
                                indices,
                                0,
                                1,
                                8,
                                pool,
                                dst_layer_ids=[0],
                            )
                        elif path == "dcp":
                            manager.send_kvcache_dcp(
                                "session",
                                indices,
                                [200],
                                indices,
                                dcp_token_item_lens=[8],
                                dst_dcp_size=1,
                                dst_dcp_rank=0,
                                src_page_offset=0,
                                decode_prefix_len=0,
                                num_kv_tokens=1,
                                executor=pool,
                                dst_layer_ids=[0],
                            )
                        else:
                            manager._transfer_data("session", [(100, 200, 8)])
                        parent = ctx.thread_context.cur_slice_stack[
                            -1
                        ].span.get_span_context()
                        carrier, sources, targets, lengths = manager.engine.calls[0]
                        self.assertEqual(
                            carrier[0],
                            f"00-{parent.trace_id:032x}-{parent.span_id:016x}-{flags}",
                        )
                        self.assertEqual(carrier[1], "vendor=value")
                        self.assertEqual(
                            (sources, targets, lengths), ([100], [200], [8])
                        )
                        self.assertEqual(
                            pool.submit(
                                lambda: getattr(
                                    manager.engine.local, "carrier", ("", "")
                                )
                            ).result(),
                            ("", ""),
                        )
                        raise StopWorker()

                    manager._prefill_unique_rank = transfer
                    original = trace.get_current_span().get_span_context()
                    with self.assertRaises(StopWorker):
                        manager.transfer_worker(
                            SimpleNamespace(get=lambda: chunk), pool
                        )
                    self.assertEqual(
                        trace.get_current_span().get_span_context(), original
                    )
                    ctx.trace_req_finish()
                    if flags == "00":
                        self.assertEqual(self.exporter.get_finished_spans(), ())
                    self.exporter.clear()

    def test_untraced_chunk_does_not_inherit_ambient_request(self):
        manager = self.manager()
        chunk = SimpleNamespace(
            trace_ctx=tracing.TraceNullContext(), room=1, staging_counted=False
        )

        def transfer():
            manager._transfer_data("session", [(100, 200, 8)])
            self.assertEqual(manager.engine.calls[-1][0], ("", ""))
            raise StopWorker()

        manager._prefill_unique_rank = transfer
        with tracing.tracer.start_as_current_span("unrelated") as ambient:
            for enabled in (True, False):
                manager.enable_trace = enabled
                with self.assertRaises(StopWorker):
                    manager.transfer_worker(SimpleNamespace(get=lambda: chunk), None)
                self.assertEqual(trace.get_current_span(), ambient)

    def test_reused_executor_restores_context_after_io_error(self):
        manager = self.manager()
        with ThreadPoolExecutor(1) as pool:
            pool.submit(manager.engine.set_trace_context, "prior", "state").result()
            for flags in ("01", "00"):
                carrier = {
                    "traceparent": f"00-12345678901234567890123456789012-1234567890123456-{flags}"
                }
                pool.submit(
                    manager._transfer_data, "session", [(100, 200, 8)], carrier
                ).result()
                self.assertEqual(manager.engine.calls[-1][0][0], carrier["traceparent"])
                with patch.object(
                    manager.engine,
                    "batch_transfer_sync",
                    side_effect=RuntimeError("IO failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "IO failure"):
                        pool.submit(
                            manager._transfer_data, "session", [(100, 200, 8)], carrier
                        ).result()
                self.assertEqual(
                    pool.submit(lambda: manager.engine.local.carrier).result(),
                    ("prior", "state"),
                )

    def test_scope_legacy_disabled_empty_and_fail_open(self):
        with native_trace_scope(object(), {}):
            pass
        engine = Engine()
        engine.set_trace_context("old", "state")
        with native_trace_scope(engine, {}):
            self.assertEqual(engine.local.carrier, ("", ""))
        self.assertEqual(engine.local.carrier, ("old", "state"))
        with patch.object(tracing, "opentelemetry_initialized", False):
            with native_trace_scope(engine, {}):
                self.assertEqual(engine.local.carrier, ("old", "state"))
        for fail_on in (1, 2):
            calls = []

            def setter(*args, calls=calls, fail_on=fail_on):
                calls.append(args)
                if len(calls) == fail_on:
                    raise ValueError("tracing failure")
                return ("", "")

            target = SimpleNamespace(set_trace_context=setter)
            with self.assertRaisesRegex(RuntimeError, "IO failure"):
                with native_trace_scope(target, {}):
                    raise RuntimeError("IO failure")

    def test_nonconforming_setter_is_cleared_before_io(self):
        engine = Engine()

        def setter(parent, state):
            engine.set_trace_context(parent, state)
            return None  # Old or nonconforming third-party binding.

        target = SimpleNamespace(
            set_trace_context=setter,
            clear_trace_context=lambda: engine.set_trace_context("", ""),
        )
        with native_trace_scope(target, {"traceparent": "new"}):
            self.assertEqual(engine.local.carrier, ("", ""))


if __name__ == "__main__":
    unittest.main()
