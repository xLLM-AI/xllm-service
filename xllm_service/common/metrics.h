/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <bvar/bvar.h>
#include <bvar/multi_dimension.h>

#include "timer.h"

namespace xllm_service {

using bvar::Adder;

class AutoCounter final {
 public:
  AutoCounter(bvar::Adder<double>& counter) : counter_(counter) {}

  ~AutoCounter() {
    // add the elapsed time to the counter
    counter_ << timer_.elapsed_seconds();
  }

 private:
  // NOLINTNEXTLINE
  bvar::Adder<double>& counter_;

  // the timer
  Timer timer_;
};

}  // namespace xllm_service

// define helpful macros to hide boilerplate code
// NOLINTBEGIN(bugprone-macro-parentheses)

// define gauge (using bvar::Status for single values)
#define DEFINE_GAUGE(name, desc) bvar::Status<double> GAUGE_##name(#name, 0.0);

#define GAUGE_SET(name, value) GAUGE_##name.set_value(value);

#define GAUGE_ADD(name, value) \
  GAUGE_##name.set_value(GAUGE_##name.get_value() + value);

#define GAUGE_INC(name) GAUGE_##name.set_value(GAUGE_##name.get_value() + 1);

#define GAUGE_VALUE(name) GAUGE_##name.get_value();

// define counter (using bvar::Adder for accumulating values)
#define DEFINE_COUNTER(name, desc) bvar::Adder<double> COUNTER_##name(#name);

#define COUNTER_ADD(name, value) COUNTER_##name << (value);

#define COUNTER_INC(name) COUNTER_##name << 1;

// Declares a latency counter having a variable name based on line number.
// example: AUTO_COUNTER(a_counter_name);
#define AUTO_COUNTER(name) \
  xllm_service::AutoCounter SAFE_CONCAT(name, __LINE__)(COUNTER_##name);

// define histogram (using bvar::LatencyRecorder for latency measurements)
#define DEFINE_HISTOGRAM(name, desc) \
  bvar::LatencyRecorder HISTOGRAM_##name(#name);

// value must be int64_t
#define HISTOGRAM_OBSERVE(name, value) HISTOGRAM_##name << (value);

// define multi histogram (using bvar::MultiDimension for multi-dimensional
// measurements)
#define DEFINE_MULTI_HISTOGRAM(name, label, desc)                     \
  bvar::MultiDimension<bvar::LatencyRecorder> MULTI_HISTOGRAM_##name( \
      #name, {(label)});

#define MULTI_HISTOGRAM_OBSERVE(name, key, value)  \
  bvar::LatencyRecorder* latency_recorder_##name = \
      MULTI_HISTOGRAM_##name.get_stats({(key)});   \
  if (latency_recorder_##name) {                   \
    *latency_recorder_##name << (value);           \
  }

// declare gauge
#define DECLARE_GAUGE(name) extern bvar::Status<double> GAUGE_##name;

// declare counter
#define DECLARE_COUNTER(name) extern bvar::Adder<double> COUNTER_##name;

// declare histogram
#define DECLARE_HISTOGRAM(name) extern bvar::LatencyRecorder HISTOGRAM_##name;

// declare multi histogram
#define DECLARE_MULTI_HISTOGRAM(name) \
  extern bvar::MultiDimension<bvar::LatencyRecorder> MULTI_HISTOGRAM_##name;

// NOLINTEND(bugprone-macro-parentheses)

DECLARE_COUNTER(server_request_in_total);

DECLARE_HISTOGRAM(time_to_first_token_latency_milliseconds);
DECLARE_HISTOGRAM(inter_token_latency_milliseconds);