"""Unit tests for the mooncake control-plane wedge hardening.

Production context (kv-router PD, mooncake backend): under a client
cancellation storm every new room wedged in KVPoll.Bootstrapping because the
sole room-registration consumer could stall on inline control-plane sends
and a poison message could kill it outright. These tests pin the hardening
properties:

- control replies are queued, never sent inline by the table owners;
- enqueueing never blocks, even when the sender thread is stuck and the
  queue is full (drop + count instead);
- a malformed message is logged and skipped without killing the consumer;
- a late registration for an already-concluded room is dropped instead of
  resurrecting the room as WaitingForInput.
"""

import queue
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Callable

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    AbortNotification,
    AckTarget,
)
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


class _WedgeTestManager(MooncakeKVManager):
    """A MooncakeKVManager shell built without the engine/registration stack.

    Everything the bootstrap_thread and the abort paths touch is plain dict
    bookkeeping, so the shell is sufficient for control-flow coverage.
    """

    def __init__(self):
        # Intentionally skip MooncakeKVManager.__init__: no transfer engine,
        # no real sockets, no worker pool. Real ZMQ IO happens only against a
        # private PULL socket created by with_bind_socket().
        self._stopped = False
        self._worker_threads = []
        self.deferred_bootstrap = None
        self.server_args = SimpleNamespace(enable_pd_role_switch=False)
        self.request_status = {}
        self.transfer_infos = {}
        self.req_to_decode_prefix_len = {}
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}
        self.enable_staging = False
        self._staging_ctx = None
        self.enable_deferred_decode_kv_release = False
        self.is_mla_backend = False
        self.failure_records = {}
        # _prefill_unique_rank() inputs for the ABORT_ACK wire id.
        self.attn_tp_rank = 0
        self.attn_cp_rank = 0
        self.attn_cp_size = 1
        self.pp_rank = 0
        self.pp_size = 1
        self._control_reply_queue = queue.Queue(maxsize=8)
        self._control_replies_dropped = 0
        self.server_socket = None
        self._zmq_ctx = None

    def with_bind_socket(self) -> None:
        self._zmq_ctx = zmq.Context()
        self.server_socket = self._zmq_ctx.socket(zmq.PULL)
        self.port = self.server_socket.bind_to_random_port("tcp://127.0.0.1")
        # One persistent PUSH connection: ZMQ FIFO ordering is per-socket,
        # and several test flows depend on send order.
        self._push_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._push_socket.setsockopt(zmq.LINGER, 0)
        self._push_socket.connect(f"tcp://127.0.0.1:{self.port}")

    def push_message(self, frames) -> None:
        self._push_socket.send_multipart(frames)

    def shutdown(self) -> None:
        # Wake the consumer(s) parked in a blocking recv with a benign
        # message so the threads observe _stopped and exit instead of dying
        # on a closed socket (rooms absent from request_status are dropped by
        # the zombie gate, so room 0 is inert).
        self._stopped = True
        if getattr(self, "_push_socket", None) is not None:
            try:
                self._push_socket.send_multipart(self._registration_frames(0))
            except Exception:
                pass
        for t in self._worker_threads:
            t.join(timeout=2.0)
        if getattr(self, "_push_socket", None) is not None:
            self._push_socket.close(linger=0)
        if self.server_socket is not None:
            self.server_socket.close(linger=0)
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except zmq.ZMQError:
                pass

    def _registration_frames(self, room: int):
        return [
            str(room).encode("ascii"),
            b"127.0.0.1",
            b"12349",
            b"sess-1",
            np.array([0, 1, 2], dtype=np.int32).tobytes(),
            b"5",
            pack_int_lists([], "i"),
            b"1",  # required_dst_info_num = 1
            b"0",  # decode_prefix_len
        ]


class TestControlReplyQueue(CustomTestCase):
    def setUp(self):
        self.mgr = _WedgeTestManager()

    def tearDown(self):
        self.mgr.shutdown()

    def test_enqueue_never_blocks_when_sender_stuck_and_queue_full(self):
        release = threading.Event()

        def stuck_send(endpoint, parts, is_ipv6=False):
            release.wait(timeout=30)

        self.mgr._send_multipart_locked = stuck_send
        self.mgr._start_control_reply_thread()

        start = time.monotonic()
        for _ in range(32):  # queue maxsize is 8; everything past it must drop
            self.mgr._enqueue_control_reply(
                "tcp://127.0.0.1:9999", [ABORT_ACK_TAG, b"123"]
            )
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 1.0, "enqueue blocked on a stalled sender")
        self.assertEqual(self.mgr._control_reply_queue.qsize(), 8)
        self.assertEqual(self.mgr._control_replies_dropped, 24)
        release.set()
        self.assertTrue(
            _wait_for(lambda: self.mgr._control_reply_queue.qsize() == 0, timeout=5.0),
            "sender did not drain the queue after release",
        )

    def test_abort_ack_uses_queue_not_inline_send(self):
        sent_inline = []
        self.mgr._send_multipart_locked = lambda endpoint, parts, is_ipv6=False: (
            sent_inline.append((endpoint, parts))
        )

        target = AckTarget(ip="127.0.0.1", port=9999, generation=3)
        self.mgr._send_abort_ack(42, target)

        self.assertEqual(sent_inline, [])
        endpoint, parts, is_ipv6 = self.mgr._control_reply_queue.get_nowait()
        self.assertEqual(endpoint, "tcp://127.0.0.1:9999")
        self.assertEqual(parts[0], ABORT_ACK_TAG)
        self.assertIn(b"42", parts)

    def test_legacy_abort_notification_enqueues_ack(self):
        self.mgr._send_multipart_locked = None  # any inline send must explode
        self.mgr.request_status[42] = KVPoll.Bootstrapping

        handled = self.mgr._handle_abort_notification(
            AbortNotification(
                room=42, decode_ip="127.0.0.1", decode_port=9999, generation=7
            ).to_zmq()
        )

        self.assertTrue(handled)
        self.assertEqual(self.mgr.request_status[42], KVPoll.Failed)
        endpoint, parts, is_ipv6 = self.mgr._control_reply_queue.get_nowait()
        self.assertEqual(parts, [ABORT_ACK_TAG, b"42"])


class TestBootstrapThreadHardening(CustomTestCase):
    def setUp(self):
        self.mgr = _WedgeTestManager()
        self.mgr.with_bind_socket()
        self.mgr._send_multipart_locked = lambda endpoint, parts, is_ipv6=False: None
        self.mgr.start_prefill_thread()

    def tearDown(self):
        self.mgr.shutdown()

    def test_malformed_message_does_not_kill_consumer(self):
        # Short frames: the registration branch raises IndexError at
        # required_dst_info_num. Before the hardening this killed the thread
        # (an unowned exception in bootstrap_thread) and left every subsequent
        # room in KVPoll.Bootstrapping forever.
        self.mgr.push_message([b"123", b"127.0.0.1", b"12349", b"sess-1"])

        self.mgr.request_status[456] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(456))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(456) == KVPoll.WaitingForInput
            ),
            "consumer died on a malformed message: valid registration never processed",
        )

    def test_late_registration_dropped_without_resurrection(self):
        # Room 123 has no entry in request_status (timed out / aborted and
        # cleared). A late registration must not resurrect it. Pushing a
        # valid room-124 registration afterwards is the consumption barrier:
        # the per-socket FIFO proves 123 was processed by the time 124
        # reaches WaitingForInput.
        self.mgr.request_status[124] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(123))
        self.mgr.push_message(self.mgr._registration_frames(124))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(124) == KVPoll.WaitingForInput
            ),
            "barrier room did not bootstrap",
        )
        self.assertNotIn(123, self.mgr.request_status)
        self.assertNotIn(123, self.mgr.transfer_infos)

    def test_live_registration_still_bootstraps(self):
        self.mgr.request_status[789] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(789))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(789) == KVPoll.WaitingForInput
            ),
            "healthy room did not leave KVPoll.Bootstrapping",
        )
        self.assertIn(789, self.mgr.transfer_infos)


if __name__ == "__main__":
    unittest.main()
