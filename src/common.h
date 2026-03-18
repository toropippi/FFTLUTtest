#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fftlut {

constexpr double kPi = 3.141592653589793238462643383279502884;

inline void CheckCuda(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
            << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) ::fftlut::CheckCuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
inline T Clamp(T value, T lo, T hi) {
    return std::max(lo, std::min(value, hi));
}

inline bool IsPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

inline int IntegerLog2(int value) {
    int result = 0;
    while ((1 << result) < value) {
        ++result;
    }
    return result;
}

inline uint32_t ReverseBits(uint32_t value, int bit_count) {
    uint32_t reversed = 0;
    for (int i = 0; i < bit_count; ++i) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

inline std::string ToLower(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

inline std::string FormatDouble(double value, int precision = 10) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream oss;
    oss << std::setprecision(precision) << std::scientific << value;
    return oss.str();
}

inline std::string JsonEscape(const std::string& value) {
    std::ostringstream oss;
    for (char c : value) {
        switch (c) {
        case '\\': oss << "\\\\"; break;
        case '"': oss << "\\\""; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20U) {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
            } else {
                oss << c;
            }
            break;
        }
    }
    return oss.str();
}

inline std::string JoinPath(const std::filesystem::path& lhs, const std::string& rhs) {
    return (lhs / rhs).string();
}

inline std::vector<double> ToDoubleVector(const std::vector<float>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<double>(input[i]);
    }
    return output;
}

inline std::vector<float> RealFromComplex(const std::vector<float2>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i].x;
    }
    return output;
}

inline std::vector<float> RealFromComplex(const std::vector<std::complex<double>>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i].real());
    }
    return output;
}

inline std::vector<float> ToFloatVector(const std::vector<double>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]);
    }
    return output;
}

inline std::vector<float2> ToFloat2Vector(const std::vector<std::complex<double>>& input) {
    std::vector<float2> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = make_float2(static_cast<float>(input[i].real()), static_cast<float>(input[i].imag()));
    }
    return output;
}

struct ImagePreset {
    std::string image_type;
    int variant_id = 0;
    std::string variant_name;
};

struct RgbImageF {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
};

struct HdrSpotMetadata {
    int center_x = 0;
    int center_y = 0;
    float radius = 0.0f;
    float peak_intensity = 0.0f;
    float color_r = 0.0f;
    float color_g = 0.0f;
    float color_b = 0.0f;
};

struct CaseSpec {
    int width = 0;
    int height = 0;
    std::string image_type;
    int variant_id = 0;
    uint32_t seed = 0;
    bool save_images = true;
    bool save_spectrum = true;
};

struct MetricSet {
    double mse_vs_ref = 0.0;
    double rmse_vs_ref = 0.0;
    double mae_vs_ref = 0.0;
    double max_abs_error = 0.0;
    double psnr_vs_ref = 0.0;
    bool psnr_is_infinite = false;
    double relative_l2_error = 0.0;
    double mean_abs_error_in_spectrum = 0.0;
    double max_abs_error_in_spectrum = 0.0;
    double mse_vs_original = 0.0;
    double mae_vs_original = 0.0;
    double max_abs_error_vs_original = 0.0;
};

struct CrossMetricSet {
    double mse_fast_vs_lut = 0.0;
    double mae_fast_vs_lut = 0.0;
    double max_abs_fast_vs_lut = 0.0;
    double relative_l2_fast_vs_lut = 0.0;
};

struct ModeReport {
    std::string mode;
    MetricSet metrics;
};

struct CaseReport {
    CaseSpec spec;
    std::string variant_name;
    std::filesystem::path case_dir;
    std::vector<ModeReport> mode_reports;
    CrossMetricSet cross_metrics;
};

struct RunConfig {
    bool run_all = false;
    int width = 512;
    int height = 512;
    std::string image_type = "checkerboard";
    int variant_id = 0;
    uint32_t seed = 1234;
    std::filesystem::path output_dir = "output";
    bool save_images = true;
    bool save_spectrum = true;
};

}  // namespace fftlut
