"""CPU regression tests for the request -> Mooncake -> worker trace boundary."""

import pickle
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import sglang.srt.observability.trace as trace_mod
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationTrace(CustomTestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.patch = patch.multiple(
            trace_mod,
            opentelemetry_initialized=True,
            tracer=self.provider.get_tracer("test"),
            global_trace_modules=["request", "mooncake"],
            threads_info={},
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.provider.shutdown)

    def create_sender(self, ctx, supports_header=True):
        sender_class = Mock()
        sender_class.supports_external_trace_header = supports_header
        queue = SimpleNamespace(
            _check_if_req_exceed_kv_capacity=lambda req: False,
            transfer_backend="mooncake",
            tp_rank=0,
            pp_rank=0,
            kv_manager=object(),
            bootstrap_port=8998,
            _process_req=Mock(),
        )
        req = SimpleNamespace(
            bootstrap_host="localhost",
            bootstrap_room=123,
            disagg_prefill_dp_rank=None,
            time_stats=SimpleNamespace(trace_ctx=ctx),
        )
        with patch(
            "sglang.srt.disaggregation.prefill.get_kv_class", return_value=sender_class
        ):
            self.assertTrue(PrefillBootstrapQueue.create_sender(queue, req, 8))
        return sender_class.call_args.kwargs

    def test_sampled_and_unsampled_parent_cross_pickle_and_thread(self):
        for flags in ("01", "00"):
            with self.subTest(flags=flags):
                self.exporter.clear()
                request = TraceReqContext(
                    "request",
                    external_trace_header={
                        "traceparent": f"00-12345678901234567890123456789012-1234567890123456-{flags}",
                        "tracestate": "vendor=value",
                    },
                )
                request.trace_req_start()
                request_copy = pickle.loads(pickle.dumps(request))
                kwargs = self.create_sender(request_copy)
                carrier = kwargs["external_trace_header"]
                self.assertEqual(carrier["traceparent"].split("-")[-1], flags)
                self.assertEqual(carrier["tracestate"], "vendor=value")
                sender = object.__new__(MooncakeKVSender)
                sender.kv_mgr = SimpleNamespace(enable_trace=True)
                sender.bootstrap_room = 123
                sender._init_trace_ctx(carrier)
                sender_root = trace.get_current_span(
                    sender.trace_ctx.root_span_context
                ).get_span_context()
                parent = trace.get_current_span(
                    request.root_span_context
                ).get_span_context()
                self.assertEqual(sender_root.trace_id, parent.trace_id)
                self.assertEqual(sender_root.trace_flags, parent.trace_flags)
                worker_ctx = sender.trace_ctx.copy_for_thread()

                def run_worker():
                    worker_ctx.rebuild_thread_context()
                    worker_ctx.trace_slice_start("transfer", 1)
                    worker_ctx.trace_slice_end("transfer", 1, thread_finish_flag=True)

                worker = threading.Thread(target=run_worker)
                worker.start()
                worker.join()
                self.assertIsNone(worker_ctx.thread_context)
                sender.trace_ctx.trace_req_finish()
                request.trace_req_finish()
                spans = self.exporter.get_finished_spans()
                if flags == "00":
                    self.assertEqual(spans, ())
                else:
                    self.assertEqual(
                        {s.context.trace_id for s in spans}, {parent.trace_id}
                    )
                    sender_span = next(
                        s for s in spans if s.name.startswith("Sender Req")
                    )
                    self.assertEqual(sender_span.parent.span_id, parent.span_id)
                    transfer_span = next(s for s in spans if s.name == "transfer")
                    worker_span = next(
                        s
                        for s in spans
                        if s.context.span_id == transfer_span.parent.span_id
                    )
                    self.assertEqual(worker_span.parent.span_id, sender_root.span_id)

    def test_disabled_and_other_backends_do_not_receive_header(self):
        self.assertNotIn(
            "external_trace_header", self.create_sender(TraceNullContext())
        )
        request = TraceReqContext("request")
        request.trace_req_start()
        self.assertNotIn("external_trace_header", self.create_sender(request, False))
        request.trace_req_finish()

    def test_sender_uses_w3c_with_non_w3c_global_propagator(self):
        original = propagate.get_global_textmap()
        global_propagator = Mock()
        global_propagator.inject.side_effect = lambda carrier, **kwargs: carrier.update(
            {"b3": "non-w3c-context"}
        )
        propagate.set_global_textmap(global_propagator)
        self.addCleanup(propagate.set_global_textmap, original)
        request = TraceReqContext("request")
        request.trace_req_start()
        self.addCleanup(request.trace_req_finish)
        carrier = self.create_sender(request)["external_trace_header"]
        self.assertIn("traceparent", carrier)
        self.assertNotIn("b3", carrier)
        global_propagator.inject.assert_not_called()
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = SimpleNamespace(enable_trace=True)
        sender.bootstrap_room = 123
        sender._init_trace_ctx(carrier)
        self.addCleanup(sender.trace_ctx.trace_req_finish)
        parent = trace.get_current_span(request.root_span_context).get_span_context()
        self.assertEqual(sender.trace_ctx.root_span.parent.span_id, parent.span_id)
        self.assertEqual(
            sender.trace_ctx.root_span.get_span_context().trace_id, parent.trace_id
        )


if __name__ == "__main__":
    unittest.main()
