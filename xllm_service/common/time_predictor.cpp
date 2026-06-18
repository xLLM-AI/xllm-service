/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "time_predictor.h"

namespace {
constexpr int32_t kDegree = 2;
}  // namespace

namespace xllm_service {

TimePredictor::TimePredictor(
    const std::vector<std::pair<int32_t, double>>& ttft_profiling_data,
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        tpot_profiling_data) {
  if (!ttft_profiling_data.empty()) {
    // construct Vandermonde matrix
    int32_t m = ttft_profiling_data.size();
    int32_t n = kDegree + 1;
    Eigen::MatrixXd matrix(m, n);
    for (int32_t i = 0; i < m; ++i) {
      for (int32_t j = 0; j < n; ++j) {
        matrix(i, j) = std::pow(ttft_profiling_data[i].first, j);
      }
    }

    // construct target vector
    Eigen::VectorXd target(m);
    for (int32_t i = 0; i < m; ++i) {
      target(i) = ttft_profiling_data[i].second;
    }

    // get coefficients
    ttft_coefficients_ = matrix.colPivHouseholderQr().solve(target);
  } else {
    ttft_coefficients_ = Eigen::VectorXd::Zero(1);
  }

  if (!tpot_profiling_data.empty()) {
    int32_t m = tpot_profiling_data.size();
    int32_t n = kDegree + 1;
    Eigen::MatrixXd matrix(m, n);
    for (int32_t i = 0; i < m; ++i) {
      int32_t avg_length = std::get<0>(tpot_profiling_data[i]);
      int32_t batch_size = std::get<1>(tpot_profiling_data[i]);

      matrix(i, 0) = 1.0;  // the index 0 is always for constant
      matrix(i, 1) = batch_size;
      matrix(i, 2) = batch_size * (avg_length - 1);
    }

    // construct target vector
    Eigen::VectorXd target(m);
    for (int32_t i = 0; i < m; ++i) {
      target(i) = std::get<2>(tpot_profiling_data[i]);
    }

    // get coefficients
    tpot_coefficients_ = matrix.colPivHouseholderQr().solve(target);
  } else {
    tpot_coefficients_ = Eigen::VectorXd::Zero(3);
  }
}

double TimePredictor::predict_ttft(int32_t length) {
  double result = 0.0;
  double power = 1.0;
  for (int32_t i = 0; i < ttft_coefficients_.size(); ++i) {
    result += ttft_coefficients_(i) * power;
    power *= length;
  }

  return result;
}

double TimePredictor::predict_tpot(int32_t total_length, int32_t batch_size) {
  double result = 0.0;
  result = tpot_coefficients_(0) + tpot_coefficients_(1) * batch_size +
           tpot_coefficients_(2) * total_length;
  return result;
}

}  // namespace xllm_service