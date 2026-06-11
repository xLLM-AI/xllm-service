# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
"""Unit tests for vLLM /metrics parsing (no HTTP needed)."""

from vllm_sidecar.metrics import VllmMetricsScraper, _parse_samples

SAMPLE = """\
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen2.5-7b"} 4.0
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="qwen2.5-7b"} 0.37
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="qwen2.5-7b"} 2.0
vllm:time_to_first_token_seconds_count{model_name="qwen2.5-7b"} 10.0
vllm:time_per_output_token_seconds_sum{model_name="qwen2.5-7b"} 1.0
vllm:time_per_output_token_seconds_count{model_name="qwen2.5-7b"} 100.0
"""


def test_parse_samples_strips_labels_and_comments() -> None:
    s = _parse_samples(SAMPLE)
    assert s["vllm:num_requests_waiting"] == 4.0
    assert s["vllm:gpu_cache_usage_perc"] == 0.37
    assert "vllm:num_requests_waiting" in s
    # HELP/TYPE comment lines must not appear
    assert all(not k.startswith("#") for k in s)


def test_parse_samples_sums_duplicate_label_sets() -> None:
    text = (
        'vllm:num_requests_waiting{m="a"} 2.0\nvllm:num_requests_waiting{m="b"} 3.0\n'
    )
    assert _parse_samples(text)["vllm:num_requests_waiting"] == 5.0


def test_parse_samples_handles_scientific_notation() -> None:
    # Small latency values are emitted in scientific notation with a negative
    # exponent; the value must parse fully, not be truncated at "1.23e" -> 0.
    text = 'vllm:time_to_first_token_seconds_sum{m="a"} 1.23e-04\n'
    assert _parse_samples(text)["vllm:time_to_first_token_seconds_sum"] == 1.23e-04


def test_parse_samples_drops_nan_and_inf() -> None:
    # vLLM may emit NaN/Inf; these must be dropped so downstream int()/JSON
    # cannot crash the scrape loop.
    text = (
        "vllm:num_requests_waiting NaN\n"
        "vllm:gpu_cache_usage_perc +Inf\n"
        "vllm:other 2.0\n"
    )
    s = _parse_samples(text)
    assert "vllm:num_requests_waiting" not in s
    assert "vllm:gpu_cache_usage_perc" not in s
    assert s["vllm:other"] == 2.0


def test_interval_avg_ms_uses_delta() -> None:
    sc = VllmMetricsScraper("http://unused")
    # first call establishes the baseline -> 0
    assert sc._interval_avg_ms({"h_sum": 2.0, "h_count": 10.0}, "h") == 0
    # next: delta_sum=3.0 over delta_count=5 -> 0.6s avg -> 600 ms
    assert sc._interval_avg_ms({"h_sum": 5.0, "h_count": 15.0}, "h") == 600


def test_interval_avg_ms_zero_when_no_new_samples() -> None:
    sc = VllmMetricsScraper("http://unused")
    sc._interval_avg_ms({"h_sum": 2.0, "h_count": 10.0}, "h")
    assert sc._interval_avg_ms({"h_sum": 2.0, "h_count": 10.0}, "h") == 0
