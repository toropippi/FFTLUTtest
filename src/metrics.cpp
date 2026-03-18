#include "metrics.h"

namespace fftlut {

namespace {

struct ErrorAccumulator {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    double ref_sq = 0.0;
};

template <typename RefValue, typename TestValue, typename RefAccessor, typename TestAccessor>
ErrorAccumulator AccumulateImageErrors(
    const RefValue& reference,
    const TestValue& test,
    const RefAccessor& ref_accessor,
    const TestAccessor& test_accessor) {
    if (reference.size() != test.size()) {
        throw std::runtime_error("Mismatched image sizes for metrics");
    }
    ErrorAccumulator acc;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref_value = ref_accessor(reference[i]);
        const double test_value = test_accessor(test[i]);
        const double diff = test_value - ref_value;
        const double abs_diff = std::abs(diff);
        acc.sum_sq += diff * diff;
        acc.sum_abs += abs_diff;
        acc.max_abs = std::max(acc.max_abs, abs_diff);
        acc.ref_sq += ref_value * ref_value;
    }
    return acc;
}

double ComputeMeanSpectrumAbsError(
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<std::complex<double>>& spectrum,
    double* max_error) {
    if (reference_spectrum.size() != spectrum.size()) {
        throw std::runtime_error("Mismatched spectrum sizes for metrics");
    }
    double sum_abs = 0.0;
    double local_max = 0.0;
    for (size_t i = 0; i < reference_spectrum.size(); ++i) {
        const double error = std::abs(spectrum[i] - reference_spectrum[i]);
        sum_abs += error;
        local_max = std::max(local_max, error);
    }
    *max_error = local_max;
    return sum_abs / static_cast<double>(reference_spectrum.size());
}

std::vector<std::complex<double>> ToComplexDouble(const std::vector<float2>& spectrum) {
    std::vector<std::complex<double>> converted(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        converted[i] = std::complex<double>(static_cast<double>(spectrum[i].x), static_cast<double>(spectrum[i].y));
    }
    return converted;
}

MetricSet FinalizeMetricSet(
    const ErrorAccumulator& ref_acc,
    const ErrorAccumulator& original_acc,
    size_t element_count,
    double mean_spectrum_error,
    double max_spectrum_error) {
    MetricSet metrics;
    const double count = static_cast<double>(element_count);
    metrics.mse_vs_ref = ref_acc.sum_sq / count;
    metrics.rmse_vs_ref = std::sqrt(metrics.mse_vs_ref);
    metrics.mae_vs_ref = ref_acc.sum_abs / count;
    metrics.max_abs_error = ref_acc.max_abs;
    metrics.relative_l2_error = std::sqrt(ref_acc.sum_sq / std::max(ref_acc.ref_sq, 1e-30));
    metrics.mean_abs_error_in_spectrum = mean_spectrum_error;
    metrics.max_abs_error_in_spectrum = max_spectrum_error;
    if (metrics.mse_vs_ref == 0.0) {
        metrics.psnr_vs_ref = 0.0;
        metrics.psnr_is_infinite = true;
    } else {
        metrics.psnr_vs_ref = 10.0 * std::log10(1.0 / metrics.mse_vs_ref);
    }
    metrics.mse_vs_original = original_acc.sum_sq / count;
    metrics.mae_vs_original = original_acc.sum_abs / count;
    metrics.max_abs_error_vs_original = original_acc.max_abs;
    return metrics;
}

template <typename RefType, typename TestType, typename RefAccessor, typename TestAccessor>
std::vector<float> MakeAbsDiffImage(
    const RefType& a,
    const TestType& b,
    const RefAccessor& a_accessor,
    const TestAccessor& b_accessor) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Mismatched array sizes for abs diff");
    }
    std::vector<float> diff(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        diff[i] = static_cast<float>(std::abs(a_accessor(a[i]) - b_accessor(b[i])));
    }
    return diff;
}

}  // namespace

MetricSet ComputeMetricSet(
    const std::vector<double>& reference_reconstruction,
    const std::vector<double>& reconstruction,
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<std::complex<double>>& spectrum,
    const std::vector<float>& original) {
    const ErrorAccumulator ref_errors = AccumulateImageErrors(
        reference_reconstruction,
        reconstruction,
        [](double value) { return value; },
        [](double value) { return value; });
    const ErrorAccumulator original_errors = AccumulateImageErrors(
        original,
        reconstruction,
        [](float value) { return static_cast<double>(value); },
        [](double value) { return value; });
    double max_spectrum_error = 0.0;
    const double mean_spectrum_error =
        ComputeMeanSpectrumAbsError(reference_spectrum, spectrum, &max_spectrum_error);
    return FinalizeMetricSet(
        ref_errors,
        original_errors,
        reference_reconstruction.size(),
        mean_spectrum_error,
        max_spectrum_error);
}

MetricSet ComputeMetricSet(
    const std::vector<double>& reference_reconstruction,
    const std::vector<float>& reconstruction,
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<float2>& spectrum,
    const std::vector<float>& original) {
    const ErrorAccumulator ref_errors = AccumulateImageErrors(
        reference_reconstruction,
        reconstruction,
        [](double value) { return value; },
        [](float value) { return static_cast<double>(value); });
    const ErrorAccumulator original_errors = AccumulateImageErrors(
        original,
        reconstruction,
        [](float value) { return static_cast<double>(value); },
        [](float value) { return static_cast<double>(value); });
    const std::vector<std::complex<double>> spectrum_as_double = ToComplexDouble(spectrum);
    double max_spectrum_error = 0.0;
    const double mean_spectrum_error =
        ComputeMeanSpectrumAbsError(reference_spectrum, spectrum_as_double, &max_spectrum_error);
    return FinalizeMetricSet(
        ref_errors,
        original_errors,
        reference_reconstruction.size(),
        mean_spectrum_error,
        max_spectrum_error);
}

CrossMetricSet ComputeCrossMetrics(
    const std::vector<float>& gpu_fast_reconstruction,
    const std::vector<float>& gpu_lut_reconstruction) {
    const ErrorAccumulator acc = AccumulateImageErrors(
        gpu_lut_reconstruction,
        gpu_fast_reconstruction,
        [](float value) { return static_cast<double>(value); },
        [](float value) { return static_cast<double>(value); });
    CrossMetricSet metrics;
    const double count = static_cast<double>(gpu_fast_reconstruction.size());
    metrics.mse_fast_vs_lut = acc.sum_sq / count;
    metrics.mae_fast_vs_lut = acc.sum_abs / count;
    metrics.max_abs_fast_vs_lut = acc.max_abs;
    metrics.relative_l2_fast_vs_lut = std::sqrt(acc.sum_sq / std::max(acc.ref_sq, 1e-30));
    return metrics;
}

std::vector<float> ComputeAbsDiffImage(const std::vector<double>& a, const std::vector<double>& b) {
    return MakeAbsDiffImage(
        a,
        b,
        [](double value) { return value; },
        [](double value) { return value; });
}

std::vector<float> ComputeAbsDiffImage(const std::vector<double>& a, const std::vector<float>& b) {
    return MakeAbsDiffImage(
        a,
        b,
        [](double value) { return value; },
        [](float value) { return static_cast<double>(value); });
}

std::vector<float> ComputeAbsDiffImage(const std::vector<float>& a, const std::vector<float>& b) {
    return MakeAbsDiffImage(
        a,
        b,
        [](float value) { return static_cast<double>(value); },
        [](float value) { return static_cast<double>(value); });
}

std::vector<float> ComputeSpectrumAbsDiff(
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<std::complex<double>>& spectrum) {
    if (reference_spectrum.size() != spectrum.size()) {
        throw std::runtime_error("Mismatched spectrum sizes for abs diff");
    }
    std::vector<float> diff(reference_spectrum.size());
    for (size_t i = 0; i < reference_spectrum.size(); ++i) {
        diff[i] = static_cast<float>(std::abs(spectrum[i] - reference_spectrum[i]));
    }
    return diff;
}

std::vector<float> ComputeSpectrumAbsDiff(
    const std::vector<std::complex<double>>& reference_spectrum,
    const std::vector<float2>& spectrum) {
    if (reference_spectrum.size() != spectrum.size()) {
        throw std::runtime_error("Mismatched spectrum sizes for abs diff");
    }
    std::vector<float> diff(reference_spectrum.size());
    for (size_t i = 0; i < reference_spectrum.size(); ++i) {
        const std::complex<double> value(
            static_cast<double>(spectrum[i].x),
            static_cast<double>(spectrum[i].y));
        diff[i] = static_cast<float>(std::abs(value - reference_spectrum[i]));
    }
    return diff;
}

}  // namespace fftlut
