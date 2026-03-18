#pragma once

#include "common.h"

namespace fftlut {

MetricSet ComputeMetricSet(
    const std::vector<double>& reference_reconstruction,
    const std::vector<double>& reconstruction,
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<std::complex<double>>& spectrum,
    const std::vector<float>& original);

MetricSet ComputeMetricSet(
    const std::vector<double>& reference_reconstruction,
    const std::vector<float>& reconstruction,
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<float2>& spectrum,
    const std::vector<float>& original);

CrossMetricSet ComputeCrossMetrics(
    const std::vector<float>& gpu_fast_reconstruction,
    const std::vector<float>& gpu_lut_reconstruction);

std::vector<float> ComputeAbsDiffImage(const std::vector<double>& a, const std::vector<double>& b);
std::vector<float> ComputeAbsDiffImage(const std::vector<double>& a, const std::vector<float>& b);
std::vector<float> ComputeAbsDiffImage(const std::vector<float>& a, const std::vector<float>& b);

std::vector<float> ComputeSpectrumAbsDiff(
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<std::complex<double>>& spectrum);

std::vector<float> ComputeSpectrumAbsDiff(
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<float2>& spectrum);

}  // namespace fftlut
