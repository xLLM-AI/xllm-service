# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Scrape vLLM's Prometheus `/metrics` into the xllm-service heartbeat schema.

Maps vLLM gauges/histograms onto proto HeartbeatRequest fields (see
proto/xllm_rpc_service.proto LoadMetrics / LatencyMetrics):

    vllm:num_requests_waiting        -> load_metrics.waiting_requests_num
    vllm:gpu_cache_usage_perc        -> load_metrics.gpu_cache_usage_perc
    vllm:time_to_first_token_seconds -> latency_metrics.recent_max_ttft (ms)
    vllm:time_per_output_token_seconds -> latency_metrics.recent_max_tbt (ms)

Latency histograms only expose _sum/_count, so we report the per-interval
average (delta_sum/delta_count, in ms) as a pragmatic proxy for the proto's
"recent max" -- good enough for load-aware routing; a true max would need
bucket analysis. Metric names are pinned to vLLM 0.21.
"""

import logging
import math
import re

import requests

logger = logging.getLogger("vllm_sidecar.metrics")

# `name{labels} value [timestamp]` -> (base_name, value)
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?[0-9.eE+-]+|NaN|[-+]?Inf)\s*"
)

_WAITING = "vllm:num_requests_waiting"
_GPU_CACHE = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
_TTFT = "vllm:time_to_first_token_seconds"
_TBT = "vllm:time_per_output_token_seconds"


def _parse_samples(text: str) -> dict[str, float]:
    """Sum sample values by base metric name (ignoring labels / HELP lines)."""
    totals = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            val = float(m.group("value"))
        except ValueError:
            continue
        # Drop NaN/Inf so downstream int()/JSON on these samples can't crash.
        if not math.isfinite(val):
            continue
        totals[m.group("name")] = totals.get(m.group("name"), 0.0) + val
    return totals


class VllmMetricsScraper:
    def __init__(self, metrics_url: str, timeout: float = 3.0) -> None:
        self._url = metrics_url
        self._timeout = timeout
        self._prev = {}  # histogram base -> (sum, count) from the last scrape
        # Reuse one connection across periodic scrapes (HTTP keep-alive).
        self._session = requests.Session()

    def _interval_avg_ms(self, samples: dict[str, float], base: str) -> int:
        """Average over the interval since the last scrape, in ms (0 if none)."""
        cur = (samples.get(base + "_sum", 0.0), samples.get(base + "_count", 0.0))
        prev = self._prev.get(base)
        self._prev[base] = cur
        if prev is None:
            return 0  # first observation: establish baseline, no interval yet
        d_count = cur[1] - prev[1]
        if d_count <= 0:
            return 0
        d_sum = cur[0] - prev[0]
        return max(0, int((d_sum / d_count) * 1000.0))

    def scrape(self) -> dict | None:
        """Return the heartbeat metrics dict, or None if /metrics unreachable."""
        try:
            r = self._session.get(self._url, timeout=self._timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.debug("metrics scrape failed: %s", e)
            return None

        s = _parse_samples(r.text)
        gpu_cache = next((s[n] for n in _GPU_CACHE if n in s), 0.0)
        return {
            "load_metrics": {
                "waiting_requests_num": int(s.get(_WAITING, 0.0)),
                "gpu_cache_usage_perc": float(gpu_cache),
            },
            "latency_metrics": {
                "recent_max_ttft": self._interval_avg_ms(s, _TTFT),
                "recent_max_tbt": self._interval_avg_ms(s, _TBT),
            },
        }
