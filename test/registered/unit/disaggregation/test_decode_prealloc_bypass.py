"""Head-of-line bypass in DecodePreallocQueue.pop_preallocated admission.

A decode request whose KV reservation does not fit is a per-request size
problem, not a global exhaustion signal, but the historical admission loop
`break`ed on it: one oversized request at the FIFO head froze every later
request's preallocation on that rank until it timed out. Under a cache-heavy
PD workload (millisecond prefills, hundreds of concurrent handshakes) this
turns a single large request into a whole-rank stall.

The bypass scans past unfit entries (bounded per pass by
SGLANG_DISAGGREGATION_PREALLOC_BYPASS_WINDOW) so smaller requests behind them
still preallocate. Two guardrails keep the oversized entry from starving:
the per-pass window, and an age threshold
(SGLANG_DISAGGREGATION_PREALLOC_BYPASS_STALL_TIMEOUT) after which the pass
reverts to strict FIFO so free tokens can accumulate for it.

These tests drive the real pop_preallocated loop with the reservation verdict
stubbed per queue position; the token arithmetic behind prealloc_fits is
covered elsewhere.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - CI registry not installed locally

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _new_decode_req(rid: str):
    req = SimpleNamespace(
        rid=rid,
        priority=0,
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        finished_reason=None,
        finished=MagicMock(return_value=False),
        return_logprob=False,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=ReqKvInfo(req_pool_idx=0, cache_protected_len=0),
        time_stats=MagicMock(),
        multimodal_inputs=None,
        bootstrap_room=None,
    )
    return SimpleNamespace(
        req=req,
        waiting_for_input=True,
        kv_receiver=MagicMock(),
        metadata_buffer_index=-1,
        is_rebootstrap=False,
        host_staged=False,
        prealloc_skip_since=None,
    )


class TestPreallocBypass(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_queue(self, decode_reqs, *, window=0, stall_timeout=20):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = list(decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._room_to_req = {}
        queue.num_reserved_decode_tokens = 0
        queue._num_published_destinations = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._allocatable_tokens = MagicMock(return_value=1000)
        queue._prealloc_bypass_window = window
        queue._prealloc_bypass_stall_timeout = stall_timeout

        def pre_alloc_mock(req, prefix_indices=None, prefix_len=0, total_prefix_len=0):
            return torch.arange(
                len(req.origin_input_ids) - prefix_len, dtype=torch.int64
            )

        queue._pre_alloc = MagicMock(side_effect=pre_alloc_mock)

        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 100
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_token_pool.req_to_token = torch.arange(
            8 * 16, dtype=torch.int64
        ).reshape(8, 16)

        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 100
        queue.req_to_metadata_buffer_idx_allocator.alloc.side_effect = iter(range(100))

        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.token_to_kv_pool_allocator.page_size = 1
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue.token_to_kv_pool = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.tree_cache = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_priority_scheduling = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.waiting_queue = []
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler
        return queue

    @staticmethod
    def _install_fits(queue, unfit_rids):
        """Key the reservation verdict on the rid of the entry being examined.

        pop_preallocated walks self.queue in order within a pass and calls
        fits at most once per waiting entry, and entries are only removed
        after the whole pass, so the n-th fits call examines the n-th queue
        entry.
        """
        entries = list(queue.queue)
        calls = {"n": 0}

        def fits(
            full_tokens,
            swa_tokens,
            *,
            full_allocatable_tokens,
            swa_allocatable_tokens,
        ):
            idx = calls["n"]
            calls["n"] += 1
            rid = entries[idx].req.rid if idx < len(entries) else None
            return rid not in unfit_rids

        queue._prealloc_reservation_fits = MagicMock(side_effect=fits)
        return calls

    def test_default_off_keeps_strict_fifo(self):
        big, small = _new_decode_req("big"), _new_decode_req("small")
        queue = self._new_queue([big, small], window=0)
        self._install_fits(queue, unfit_rids={"big"})

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [big, small])

    def test_bypass_admits_small_request_behind_unfit_head(self):
        big, small = _new_decode_req("big"), _new_decode_req("small")
        queue = self._new_queue([big, small], window=4)
        self._install_fits(queue, unfit_rids={"big"})

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual([decode_req.req.rid for decode_req in preallocated], ["small"])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [big])
        # The skipped entry started aging.
        self.assertIsNotNone(big.prealloc_skip_since)

    def test_bypass_window_bounds_the_scan(self):
        big = _new_decode_req("big")
        unfit1 = _new_decode_req("unfit1")
        unfit2 = _new_decode_req("unfit2")
        small = _new_decode_req("small")
        queue = self._new_queue([big, unfit1, unfit2, small], window=2)
        self._install_fits(queue, unfit_rids={"big", "unfit1", "unfit2"})

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        # Window 2 covers unfit1 and unfit2; the third skip is refused, so the
        # pass ends before reaching `small`.
        self.assertEqual(preallocated, [])
        self.assertEqual(queue.queue, [big, unfit1, unfit2, small])

    def test_aged_unfit_entry_reverts_to_strict_fifo(self):
        big, small = _new_decode_req("big"), _new_decode_req("small")
        # `big` has been unfit for 100s: far past the 20s stall timeout.
        big.prealloc_skip_since = time.monotonic() - 100
        queue = self._new_queue([big, small], window=4, stall_timeout=20)
        self._install_fits(queue, unfit_rids={"big"})

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(queue.queue, [big, small])


if __name__ == "__main__":
    unittest.main()
