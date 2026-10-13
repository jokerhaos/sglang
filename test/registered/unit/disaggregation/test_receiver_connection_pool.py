"""Unit tests for srt/disaggregation/common/conn — receiver connection_pool invalidation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, CommonKVSender
from sglang.test.test_utils import CustomTestCase


class TestKVFailureEventMetrics(CustomTestCase):
    def test_timeout_events_preserve_terminal_behavior(self):
        for cls, method, event, threshold in [
            (CommonKVSender, "_check_bootstrap_timeout", "bootstrap_timeout", "bootstrap_timeout"),
            (CommonKVReceiver, "_check_waiting_timeout", "transfer_timeout", "waiting_timeout"),
        ]:
            with self.subTest(event=event), patch(
                "sglang.srt.disaggregation.common.conn.KV_FAILURE_EVENTS"
            ) as events, patch(
                "sglang.srt.disaggregation.common.conn.time.time", return_value=100
            ):
                receiver = SimpleNamespace(
                    init_time=95, bootstrap_room=7, abort_notified=True,
                    kv_mgr=SimpleNamespace(**{threshold: 10}, record_failure=Mock(), update_status=Mock()),
                    invalidate_cached_bootstrap_infos=Mock(),
                )
                self.assertIsNone(getattr(cls, method)(receiver))
                events.labels.assert_not_called()
                receiver.init_time = 80
                self.assertEqual(getattr(cls, method)(receiver), KVPoll.Failed)
                events.labels.assert_called_once_with(event=event)
                events.labels.return_value.inc.assert_called_once_with()
                receiver.kv_mgr.update_status.assert_called_once_with(7, KVPoll.Failed)
                getattr(cls, method)(receiver)
                events.labels.return_value.inc.assert_called_once_with()

    def test_abort_is_an_event_not_a_peer_metric_label(self):
        for cls in (CommonKVSender, CommonKVReceiver):
            with self.subTest(cls=cls), patch(
                "sglang.srt.disaggregation.common.conn.KV_FAILURE_EVENTS"
            ) as events:
                receiver = SimpleNamespace(
                    bootstrap_room=7, abort_notified=True,
                    kv_mgr=SimpleNamespace(record_failure=Mock(), update_status=Mock()),
                    # v0.5.21 abort() now ends with ensure_abort_notified();
                    # the event metric test does not exercise the notify path.
                    ensure_abort_notified=lambda **kw: None,
                )
                cls.abort(receiver)
                events.labels.assert_called_once_with(event="abort")
                self.assertEqual(receiver.conclude_state, KVPoll.Failed)
                cls.abort(receiver)
                events.labels.return_value.inc.assert_called_once_with()


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _receiver(connection_pool, entries):
    receiver = object.__new__(_ConcreteReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(_ConcreteReceiver):
    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        self.fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self):
        return True


def _fetching_receiver(connection_pool):
    receiver = object.__new__(_FetchingReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        is_mla_backend=False,
    )
    receiver.bootstrap_addr = "prefill:8998"
    receiver.bootstrap_room = 1
    receiver.prefill_dp_rank = 0
    receiver.target_cp_ranks = [0]
    receiver.target_tp_rank = 0
    receiver.target_tp_ranks = [0]
    receiver.target_pp_ranks = [0]
    receiver._connection_pool_entries = {}
    receiver.fetch_count = 0
    return receiver


class TestReceiverConnectionPool(CustomTestCase):
    def test_invalidate_removes_matching_generation(self):
        stale = [
            {"rank_ip": "10.0.0.1", "rank_port": 1001},
            {"rank_ip": "10.0.0.1", "rank_port": 1002},
        ]
        retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
        receiver = _receiver(
            {"stale": stale, "retained": retained},
            {"stale": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"retained": retained})
        self.assertEqual(receiver._connection_pool_entries, {})

    def test_invalidate_preserves_concurrent_replacement_generation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
        receiver = _receiver(
            {"key": replacement},
            {"key": stale},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {"key": replacement})

    def test_invalidate_removes_all_matching_cp_entries(self):
        stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
        receiver = _receiver(
            {"cp0": stale_cp0, "cp1": stale_cp1},
            {"cp0": stale_cp0, "cp1": stale_cp1},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})

    def test_next_receiver_refetches_after_invalidation(self):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        connection_pool = {"prefill:8998_0_0_0": stale}
        stale_receiver = _receiver(
            connection_pool,
            {"prefill:8998_0_0_0": stale},
        )
        stale_receiver.invalidate_cached_bootstrap_infos()

        receiver = _fetching_receiver(connection_pool)
        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.fetch_count, 1)
        self.assertEqual(receiver.bootstrap_infos[0]["rank_port"], 2001)
        self.assertIs(
            connection_pool["prefill:8998_0_0_0"],
            receiver._connection_pool_entries["prefill:8998_0_0_0"],
        )

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
    def test_waiting_timeout_invalidates_cached_generation(self, _mock_time):
        stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
        receiver = _receiver({"key": stale}, {"key": stale})
        receiver.bootstrap_room = 1
        receiver.bootstrap_infos = stale
        receiver.init_time = 1.0
        receiver.abort_notified = True
        receiver.kv_mgr.waiting_timeout = 1.0
        receiver.kv_mgr.record_failure = Mock()
        receiver.kv_mgr.update_status = Mock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertEqual(receiver.kv_mgr.connection_pool, {})


if __name__ == "__main__":
    unittest.main()
