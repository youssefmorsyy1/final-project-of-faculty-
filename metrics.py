import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any

logging.getLogger("metrics").setLevel(logging.INFO)


@dataclass
class MetricsTracker:
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "matches_processed":      0,
        "total_events":           0,
        "failed_events":          0,
        "total_time":             0,
        "matches_by_competition": defaultdict(int),
        "events_by_match":        defaultdict(int),
        "failed_by_match":        defaultdict(int),
        "completions":            0,
        "failures":               0,
    })

    match_start_times: Dict[str, float] = field(default_factory=dict)
    _current_competition: str = field(default="", init=False, repr=False)

    def start(self):
        self.metrics["start_time"] = time.time()

    def finish(self):
        end_time = time.time()
        self.metrics["total_time"] = end_time - self.metrics.get("start_time", end_time)
        self.metrics["end_time"]   = end_time

    def match_processed(self, match_id):
        self.metrics["matches_processed"] += 1
        self.metrics["matches_by_competition"][self._current_competition] += 1

    def record_event(self):
        self.metrics["total_events"] += 1

    def record_failure(self):
        self.metrics["failed_events"] += 1

    def competition_completed(self, comp_name, matches_count, events, failed, duration):
        self.metrics["completions"] += 1
        self._current_competition = comp_name
        self.metrics["matches_by_competition"][comp_name] = matches_count
        self.metrics["total_events"]  += events
        self.metrics["failed_events"] += failed

    def competition_failed(self, comp_name, error):
        self.metrics["failures"] += 1
        self.metrics["failures_reason"] = f"{comp_name}: {error}"

    def get_metrics(self) -> Dict[str, Any]:
        m = dict(self.metrics)
        for key in ("matches_by_competition", "events_by_match", "failed_by_match"):
            if key in m:
                m[key] = dict(m[key])

        processed = m.get("matches_processed", 0) or 1
        total_ev  = m.get("total_events", 0)

        m["events_per_match"]   = round(total_ev / processed, 2)
        m["avg_time_per_match"] = round(m.get("total_time", 0) / processed, 3)
        m["success_rate"]       = round(
            (1 - m.get("failed_events", 0) / max(1, total_ev)) * 100, 2
        ) if total_ev > 0 else 100.0

        return m