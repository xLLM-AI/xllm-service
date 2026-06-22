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

#include "metrics.h"

DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");

// ttft latency histogram
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");