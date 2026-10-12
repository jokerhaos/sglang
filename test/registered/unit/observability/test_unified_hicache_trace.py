"""CPU regression for the unified cache's storage-span terminal paths."""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability import trace as tracing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestUnifiedHiCacheTrace(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(id_generator=tracing.TraceCustomIdGenerator())
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        patcher = patch.multiple(
            tracing,
            opentelemetry_initialized=True,
            tracer=self.provider.get_tracer("unified-hicache-test"),
            global_trace_modules=["hicache"],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(tracing, "get_global_trace_level", return_value=1)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.provider.shutdown)
        self.carrier = {
            "traceparent": "00-12345678901234567890123456789012-1234567890123456-01"
        }

    def make_cache(self, *, hit_pages=2, read_pages=2):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 2
        controller.storage_stop_event = threading.Event()
        for name in (
            "prefetch_queue",
            "prefetch_hit_queue",
            "prefetch_buffer",
            "prefetch_sync_queue",
            "ack_prefetch_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            setattr(controller, name, Queue())
        controller.ack_write_queue = []
        controller.ack_load_queue = []
        controller.extra_host_mem_release_queues = {}
        controller.prefetch_hits_sync_groups = []
        controller.prefetch_completion_sync_groups = []
        controller.prefetch_tokens_occupied = 0
        controller.prefetch_capacity_limit = 16
        controller.host_memory_mode = "cache"
        controller.mem_pool_host = SimpleNamespace(
            alloc=lambda size: torch.arange(size),
            free=Mock(),
            page_size=2,
            available_size=lambda: 0,
        )
        controller.storage_backend = SimpleNamespace(
            batch_exists=lambda keys, extra_info: min(hit_pages, len(keys)),
        )
        controller.page_get_func = lambda op, keys, indices, extra: min(
            read_pages, len(keys)
        )

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            page_size=2,
            is_eagle=False,
            enable_storage=True,
            write_back_duplicate_reclaim_digest=0,
            prefetch_anchor_info=lambda node: (None, None),
            insert_host=Mock(
                return_value=SimpleNamespace(
                    prefix_len=0,
                    cache_actions=[],
                    host_insert_dropped=False,
                )
            ),
            commit_hicache_transfers=Mock(),
            reset=Mock(),
            kv_events=SimpleNamespace(record_all_cleared=Mock()),
        )
        cache.cache_controller = controller
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = cache.linker = None
        cache.tree_components = []
        cache.prefetch_threshold = 2
        cache.prefetch_stop_policy = "timeout"
        cache.prefetch_timeout_base = cache.prefetch_timeout_per_page = 0
        cache.pp_rank = 0
        cache.pp_size = cache.tp_world_size = 1
        cache.attn_cp_group = cache.attn_tp_group = None
        cache.work_list = []
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.inc_host_lock_ref = Mock(
            return_value=SimpleNamespace(
                to_dec_params=lambda: object(),
            )
        )
        cache.dec_host_lock_ref = Mock()
        cache.evict_host = Mock()
        cache.storage_existence_cache = StorageExistenceCache()
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache._apply_cache_actions = Mock()
        handle = CacheRequestHandle("unified-trace", 0, self.carrier)
        cache.prefetch_from_storage(handle, 0, [1, 2, 3, 4])
        return cache, controller, handle

    def run_worker(self, controller, worker, output, complete):
        controller.storage_stop_event.clear()
        produced = []
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(worker)
            try:
                while True:
                    item = output.get(timeout=5)
                    produced.append(item)
                    if complete(item):
                        break
            finally:
                controller.storage_stop_event.set()
                future.result(timeout=5)
        for item in produced:
            output.put(item)

    def query(self, cache, controller):
        self.run_worker(
            controller,
            controller.prefetch_thread_func,
            controller.prefetch_hit_queue,
            lambda operation: True,
        )
        cache.check_hicache_events()

    def read(self, controller):
        self.run_worker(
            controller,
            controller.prefetch_io_aux_func,
            controller.prefetch_sync_queue,
            lambda ack: ack.completed_req,
        )
        self.run_worker(
            controller,
            controller.prefetch_sync_thread_func,
            controller.ack_prefetch_queue,
            lambda ack: ack.completed_req,
        )

    def assert_terminal(self, outcome, tokens, *, read=True, query=True):
        spans = self.exporter.get_finished_spans()
        names = {"hicache.prefetch", "hicache.wait"}
        if query:
            names.add("hicache.query")
        if read:
            names.add("hicache.read")
        self.assertEqual(len(spans), len(names))
        self.assertEqual({s.name for s in spans}, names)
        parent = next(s for s in spans if s.name == "hicache.prefetch")
        self.assertEqual(parent.parent.span_id, int("1234567890123456", 16))
        self.assertEqual(parent.attributes["hicache.outcome"], outcome)
        self.assertEqual(parent.attributes["hicache.completed_tokens"], tokens)
        for span in spans:
            self.assertEqual(span.context.trace_id, parent.context.trace_id)
            if span is not parent:
                self.assertEqual(span.parent.span_id, parent.context.span_id)

    def test_workers_ack_and_unified_scheduler_close_parent(self):
        for pages, outcome in ((2, "complete"), (1, "partial"), (0, "partial")):
            with self.subTest(pages=pages):
                self.exporter.clear()
                cache, controller, handle = self.make_cache(read_pages=pages)
                self.query(cache, controller)
                self.read(controller)
                self.assertEqual(
                    {s.name for s in self.exporter.get_finished_spans()},
                    {"hicache.query", "hicache.read"},
                )
                cache.check_hicache_events()
                cache.check_hicache_events()
                self.assert_terminal(outcome, pages * 2)
                self.assertEqual(cache.pop_prefetch_loaded_tokens(handle), pages * 2)
                self.assertEqual(controller.prefetch_tokens_occupied, 0)
                self.assertEqual(cache.ongoing_prefetch, {})
                cache.dec_host_lock_ref.assert_called_once()

    def test_miss_revokes_and_closes_parent(self):
        cache, controller, _ = self.make_cache(hit_pages=0)
        self.query(cache, controller)
        cache.check_hicache_events()
        self.assert_terminal("miss", 0, read=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_below_threshold_and_allocation_failure_close_parent(self):
        for outcome in ("below_threshold", "allocation_failed"):
            with self.subTest(outcome=outcome):
                self.exporter.clear()
                cache, controller, _ = self.make_cache()
                if outcome == "below_threshold":
                    cache.prefetch_threshold = 8
                else:
                    controller.mem_pool_host.alloc = lambda size: None
                self.query(cache, controller)
                self.assert_terminal(outcome, 0, read=False)
                self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_abort_before_query_closes_parent(self):
        cache, controller, handle = self.make_cache()
        cache.release_aborted_request(handle)
        self.query(cache, controller)
        cache.release_aborted_request(handle)
        self.assert_terminal("cancelled", 0, read=False, query=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_timeout_then_late_ack_keeps_first_outcome(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        self.assertTrue(cache.check_prefetch_progress(handle))
        cache.check_hicache_events()
        cache.check_hicache_events()
        self.assert_terminal("timeout", 0)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_completed_read_is_not_mislabeled_timeout(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        operation = cache.ongoing_prefetch[handle].operation
        # Progress ACKs precede the terminal ACK; all pages are already agreed.
        operation.completed_tokens = 4
        self.assertTrue(cache.check_prefetch_progress(handle))
        cache.check_hicache_events()
        self.assert_terminal("complete", 4)

    def test_cancel_after_read_then_late_ack_closes_once(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        cache.release_aborted_request(handle)
        cache.check_hicache_events()
        cache.release_aborted_request(handle)
        self.assert_terminal("cancelled", 0)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_buffer_staging_closes_io_parent_before_admission(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        cache.host_memory_mode = "buffer_only"
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline.reset()
        pipeline.set_prefix_ctx(handle, [])
        cache.buffer_pipeline = pipeline
        cache.check_hicache_events()
        self.assert_terminal("complete", 4)
        self.assertTrue(pipeline.has_staged(handle))
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertTrue(pipeline.release_staged_hold(handle))
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_hybrid_shortfall_closes_parent_as_partial(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        operation = cache.ongoing_prefetch[handle].operation
        operation.pool_transfers = [PoolTransfer(name=PoolName.SWA, keys=["a"])]
        operation.pool_transfers_done = True
        cache.check_hicache_events()
        self.assert_terminal("partial", 0)
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_unsampled_workers_and_scheduler_do_not_export(self):
        self.carrier["traceparent"] = self.carrier["traceparent"][:-2] + "00"
        cache, controller, _ = self.make_cache()
        self.query(cache, controller)
        self.read(controller)
        cache.check_hicache_events()
        self.assertEqual(self.exporter.get_finished_spans(), ())
        self.assertEqual(cache.ongoing_prefetch, {})

    def test_io_exception_closes_parent_and_abort_does_not_overwrite_error(self):
        cache, controller, handle = self.make_cache()
        self.query(cache, controller)

        def fail_read(*args):
            raise RuntimeError("storage IO failed")

        controller.page_get_func = fail_read
        controller.storage_stop_event.clear()
        with ThreadPoolExecutor(1) as executor:
            with self.assertRaisesRegex(RuntimeError, "storage IO failed"):
                executor.submit(controller.prefetch_io_aux_func).result(timeout=5)
        cache.release_aborted_request(handle)
        self.assert_terminal("error", 0)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_query_exception_closes_parent_before_allocation(self):
        cache, controller, handle = self.make_cache()

        def fail_query(*args):
            raise RuntimeError("storage query failed")

        controller.storage_backend.batch_exists = fail_query
        with ThreadPoolExecutor(1) as executor:
            with self.assertRaisesRegex(RuntimeError, "storage query failed"):
                executor.submit(controller.prefetch_thread_func).result(timeout=5)
        cache.release_aborted_request(handle)
        self.assert_terminal("error", 0, read=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_best_effort_stop_keeps_early_stop_outcome(self):
        cache, controller, handle = self.make_cache()
        cache.prefetch_stop_policy = "best_effort"
        self.assertTrue(cache.check_prefetch_progress(handle))
        self.assert_terminal("early_stop", 0, read=False, query=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_shutdown_closes_unallocated_parent(self):
        cache, controller, _ = self.make_cache()
        StorageAttachment(cache)._release_pending_storage_ops()
        self.assert_terminal("cancelled", 0, read=False, query=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_shutdown_closes_allocated_parent(self):
        cache, controller, _ = self.make_cache()
        self.query(cache, controller)
        StorageAttachment(cache)._release_pending_storage_ops()
        self.assert_terminal("cancelled", 0, read=False)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_reset_closes_pending_parent(self):
        cache, controller, _ = self.make_cache()
        cache.session_refs = SimpleNamespace(reset=Mock())
        cache.session = SimpleNamespace(slots={})
        controller.reset = Mock()
        controller.mem_pool_host.clear = Mock()
        controller.enable_storage = True
        cache._reset_full()
        self.assert_terminal("cancelled", 0, read=False, query=False)
        self.assertEqual(cache.ongoing_prefetch, {})


if __name__ == "__main__":
    unittest.main()
