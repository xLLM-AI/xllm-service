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

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm_service {

// Predictor for predicting TTFT and TPOT
class TimePredictor final {
 public:
  TimePredictor(
      const std::vector<std::pair<int32_t, double>>& ttft_profiling_data,
      const std::vector<std::tuple<int32_t, int32_t, double>>&
          tpot_profiling_data);
  ~TimePredictor() = default;

  double predict_ttft(int32_t length);

  double predict_tpot(int32_t total_length, int32_t batch_size);

  // Milliseconds of compute remaining after reusing a causal text prefix.
  // An absent/invalid profiling fit is not a zero-cost engine.
  std::optional<double> predict_prefill_time(int32_t length,
                                             int32_t cached_tokens);

 private:
  Eigen::VectorXd ttft_coefficients_;
  Eigen::VectorXd tpot_coefficients_;
  // A separately constrained fit keeps routing costs positive and monotonic
  // without changing the existing SLO predictor's TTFT/TPOT behavior.
  Eigen::Vector3d prefill_coefficients_ = Eigen::Vector3d::Zero();
  double prefill_length_scale_ = 1.0;
  bool has_prefill_profile_ = false;
};

}  // namespace xllm_service
