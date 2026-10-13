"""An empty DP synchronization batch must not retain the last running count."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler_components import metrics_reporter
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIdleBatchMetrics(unittest.TestCase):
    def setUp(self):
        # v0.5.21 routes the idle publish through _update_decode_queue_stats(),
        # which reads the disagg config bag via get_disagg(): publish one.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        # Exercise the real result handler and metrics reporter without a model.
        self.reporter = object.__new__(metrics_reporter.SchedulerMetricsReporter)
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            grammar_manager=[],
            enable_priority_scheduling=False,
            disaggregation_mode=metrics_reporter.DisaggregationMode.DECODE,
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(update_scheduler_stats=Mock()),
                streaming_session_count=lambda: 0,
                session_held_tokens=lambda: 0,
            ),
        )
        self.reporter.stats = SimpleNamespace(
            num_running_reqs=QueueCount(total=1), gen_throughput=50
        )
        self.published = []
        self.reporter.metrics_collector = SimpleNamespace(
            last_log_time=100,
            log_stats=lambda stats: self.published.append(stats.num_running_reqs.total),
        )
        self.processor = SimpleNamespace(
            metrics_reporter=self.reporter,
            output_streamer=SimpleNamespace(_stream_output_generation=Mock()),
        )
        self.batch = SimpleNamespace(reqs=[], return_logprob=False)

    def idle_step(self, copy_done=None):
        with (
            patch.object(metrics_reporter, "ENABLE_METRICS_DEVICE_TIMER", False),
            patch.object(metrics_reporter.time, "perf_counter", return_value=101),
        ):
            SchedulerBatchResultProcessor.process_batch_result_idle(
                self.processor, self.batch, SimpleNamespace(copy_done=copy_done)
            )

    def test_idle_dp_clears_stale_running_count_without_waiting_30_seconds(self):
        copy_done = Mock()
        self.idle_step(copy_done)
        copy_done.synchronize.assert_called_once()
        self.assertEqual(self.published, [0])
        self.assertEqual(self.reporter.stats.gen_throughput, 0)
        self.processor.output_streamer._stream_output_generation.assert_called_once_with(
            [], False, is_idle_batch=True
        )

    def test_repeated_empty_batches_keep_existing_rate_limit(self):
        self.idle_step()
        self.idle_step()
        self.assertEqual(self.published, [0])

    def test_metrics_disabled_does_not_publish(self):
        self.reporter.current_scheduler_metrics_enabled = False
        self.idle_step()
        self.assertEqual(self.published, [])

    def test_next_running_batch_is_not_forced_to_zero(self):
        self.reporter.scheduler.running_batch.reqs = [SimpleNamespace()]
        self.idle_step()
        self.assertEqual(self.reporter.stats.num_running_reqs.total, 1)
        self.assertEqual(self.published, [])


if __name__ == "__main__":
    unittest.main()
