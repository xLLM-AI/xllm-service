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

#include "common/time_predictor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {
constexpr int32_t kDegree = 2;

Eigen::Vector3d fit_nonnegative_quadratic(const Eigen::MatrixXd& matrix,
                                          const Eigen::VectorXd& target) {
  Eigen::Vector3d best = Eigen::Vector3d::Zero();
  double best_error = std::numeric_limits<double>::infinity();
  // There are only three coefficients. Enumerating their active sets gives
  // the nonnegative least-squares fit without an iterative solver.
  for (int32_t mask = 1; mask < 8; ++mask) {
    std::vector<int32_t> columns;
    columns.reserve(kDegree + 1);
    for (int32_t column = 0; column <= kDegree; ++column) {
      if ((mask & (1 << column)) != 0) {
        columns.emplace_back(column);
      }
    }
    Eigen::MatrixXd active(matrix.rows(), columns.size());
    for (size_t column = 0; column < columns.size(); ++column) {
      active.col(column) = matrix.col(columns[column]);
    }
    const Eigen::VectorXd fitted = active.colPivHouseholderQr().solve(target);
    if (!fitted.allFinite() || (fitted.array() < 0).any()) {
      continue;
    }
    Eigen::Vector3d coefficients = Eigen::Vector3d::Zero();
    for (size_t column = 0; column < columns.size(); ++column) {
      coefficients(columns[column]) = fitted(column);
    }
    const double error = (matrix * coefficients - target).squaredNorm();
    if (error < best_error) {
      best_error = error;
      best = coefficients;
    }
  }
  return best;
}
}  // namespace

namespace xllm_service {

TimePredictor::TimePredictor(
    const std::vector<std::pair<int32_t, double>>& ttft_profiling_data,
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        tpot_profiling_data) {
  std::unordered_set<int32_t> lengths;
  bool valid_profile = true;
  for (const auto& [length, latency] : ttft_profiling_data) {
    valid_profile &= length > 0 && std::isfinite(latency) && latency > 0;
    lengths.emplace(length);
  }
  has_prefill_profile_ = valid_profile && lengths.size() >= kDegree + 1;
  if (has_prefill_profile_) {
    prefill_length_scale_ =
        static_cast<double>(*std::max_element(lengths.begin(), lengths.end()));
    const int32_t count = static_cast<int32_t>(ttft_profiling_data.size());
    Eigen::MatrixXd matrix(count, kDegree + 1);
    Eigen::VectorXd target(count);
    for (int32_t index = 0; index < count; ++index) {
      const auto& [length, latency] = ttft_profiling_data[index];
      const double normalized = length / prefill_length_scale_;
      matrix.row(index) << 1.0, normalized, normalized * normalized;
      target(index) = latency;
    }
    prefill_coefficients_ = fit_nonnegative_quadratic(matrix, target);
  }
  if (valid_profile && !ttft_profiling_data.empty()) {
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

std::optional<double> TimePredictor::predict_prefill_time(
    int32_t length,
    int32_t cached_tokens) {
  if (!has_prefill_profile_ || length <= 0 || cached_tokens < 0 ||
      cached_tokens >= length) {
    return std::nullopt;
  }
  const double uncached = (length - cached_tokens) / prefill_length_scale_;
  const double cached = cached_tokens / prefill_length_scale_;
  // Evaluate F(N)-F(H)+F(0) directly, avoiding subtraction of two close
  // predictions for long contexts with an almost complete prefix hit.
  const double remaining_time =
      prefill_coefficients_(0) + prefill_coefficients_(1) * uncached +
      prefill_coefficients_(2) * uncached * (uncached + 2 * cached);
  if (!std::isfinite(remaining_time) || remaining_time <= 0) {
    return std::nullopt;
  }
  return remaining_time;
}

}  // namespace xllm_service
