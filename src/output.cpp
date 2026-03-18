#define _CRT_SECURE_NO_WARNINGS

#include "output.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

namespace fftlut {

namespace {

std::vector<uint8_t> FloatToU8(const std::vector<float>& pixels) {
    std::vector<uint8_t> bytes(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        const float value = Clamp(pixels[i], 0.0f, 1.0f);
        bytes[i] = static_cast<uint8_t>(std::lround(value * 255.0f));
    }
    return bytes;
}

std::vector<uint8_t> FloatRgbToU8(const std::vector<float>& pixels) {
    std::vector<uint8_t> bytes(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        const float value = Clamp(pixels[i], 0.0f, 1.0f);
        bytes[i] = static_cast<uint8_t>(std::lround(value * 255.0f));
    }
    return bytes;
}

std::vector<float> NormalizeLinear(const std::vector<float>& pixels) {
    if (pixels.empty()) {
        return {};
    }
    const auto [min_it, max_it] = std::minmax_element(pixels.begin(), pixels.end());
    const float min_value = *min_it;
    const float max_value = *max_it;
    if (max_value <= min_value) {
        return std::vector<float>(pixels.size(), 0.0f);
    }
    std::vector<float> normalized(pixels.size());
    const float denom = max_value - min_value;
    for (size_t i = 0; i < pixels.size(); ++i) {
        normalized[i] = (pixels[i] - min_value) / denom;
    }
    return normalized;
}

std::vector<float> NormalizeLog(const std::vector<float>& pixels) {
    if (pixels.empty()) {
        return {};
    }
    const float max_value = *std::max_element(pixels.begin(), pixels.end());
    if (max_value <= 0.0f) {
        return std::vector<float>(pixels.size(), 0.0f);
    }
    const float epsilon = std::max(max_value * 1.0e-6f, 1.0e-12f);
    const float denom = std::log1pf(max_value / epsilon);
    std::vector<float> normalized(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        normalized[i] = std::log1pf(std::max(0.0f, pixels[i]) / epsilon) / denom;
    }
    return normalized;
}

std::vector<float> FftShift(const std::vector<float>& input, int width, int height) {
    std::vector<float> output(input.size(), 0.0f);
    const int half_w = width / 2;
    const int half_h = height / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int shifted_x = (x + half_w) % width;
            const int shifted_y = (y + half_h) % height;
            output[static_cast<size_t>(shifted_y) * static_cast<size_t>(width) + static_cast<size_t>(shifted_x)] =
                input[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
        }
    }
    return output;
}

void WriteJsonMetricSet(std::ostream& os, const MetricSet& metrics) {
    os << "{\n"
       << "      \"mse_vs_ref\": " << FormatDouble(metrics.mse_vs_ref) << ",\n"
       << "      \"rmse_vs_ref\": " << FormatDouble(metrics.rmse_vs_ref) << ",\n"
       << "      \"mae_vs_ref\": " << FormatDouble(metrics.mae_vs_ref) << ",\n"
       << "      \"max_abs_error\": " << FormatDouble(metrics.max_abs_error) << ",\n"
       << "      \"psnr_vs_ref\": ";
    if (metrics.psnr_is_infinite) {
        os << "null,\n";
    } else {
        os << FormatDouble(metrics.psnr_vs_ref) << ",\n";
    }
    os << "      \"psnr_vs_ref_is_infinite\": " << (metrics.psnr_is_infinite ? "true" : "false") << ",\n"
       << "      \"relative_l2_error\": " << FormatDouble(metrics.relative_l2_error) << ",\n"
       << "      \"mean_abs_error_in_spectrum\": " << FormatDouble(metrics.mean_abs_error_in_spectrum) << ",\n"
       << "      \"max_abs_error_in_spectrum\": " << FormatDouble(metrics.max_abs_error_in_spectrum) << ",\n"
       << "      \"mse_vs_original\": " << FormatDouble(metrics.mse_vs_original) << ",\n"
       << "      \"mae_vs_original\": " << FormatDouble(metrics.mae_vs_original) << ",\n"
       << "      \"max_abs_error_vs_original\": " << FormatDouble(metrics.max_abs_error_vs_original) << "\n"
       << "    }";
}

std::string CaseId(const CaseReport& report) {
    std::ostringstream oss;
    oss << "case_" << report.spec.image_type << "_" << report.spec.variant_id << "_"
        << report.spec.width << "x" << report.spec.height;
    return oss.str();
}

std::string BuildNpyHeader(const std::vector<int>& shape) {
    std::ostringstream shape_stream;
    shape_stream << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_stream << shape[i];
        if (shape.size() == 1 || i + 1 != shape.size()) {
            shape_stream << ", ";
        }
    }
    shape_stream << ")";

    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': " << shape_stream.str() << ", }";
    std::string header = header_stream.str();
    const size_t preamble = 10;
    const size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    header.append(padding, ' ');
    header.push_back('\n');
    return header;
}

}  // namespace

void EnsureDirectory(const std::filesystem::path& path) {
    std::filesystem::create_directories(path);
}

void SaveUnitFloatImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height) {
    EnsureDirectory(path.parent_path());
    const std::vector<uint8_t> bytes = FloatToU8(pixels);
    if (stbi_write_png(path.string().c_str(), width, height, 1, bytes.data(), width) == 0) {
        throw std::runtime_error("Failed to write PNG: " + path.string());
    }
}

void SaveLinearAutoImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height) {
    SaveUnitFloatImagePng(path, NormalizeLinear(pixels), width, height);
}

void SaveLogAutoImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height) {
    SaveUnitFloatImagePng(path, NormalizeLog(pixels), width, height);
}

void SaveNpyFloat32(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height) {
    SaveNpyFloat32(path, pixels, {height, width});
}

void SaveNpyFloat32(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    const std::vector<int>& shape) {
    EnsureDirectory(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open NPY output: " + path.string());
    }

    const std::string header = BuildNpyHeader(shape);

    output.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    output.write(version, 2);
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    output.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size() * sizeof(float)));
}

void SaveHdrRgbNpy(const std::filesystem::path& path, const RgbImageF& image) {
    SaveNpyFloat32(path, image.pixels, {image.height, image.width, 3});
}

float ComputePercentile(const std::vector<float>& values, float percentile) {
    if (values.empty()) {
        return 0.0f;
    }
    std::vector<float> filtered;
    filtered.reserve(values.size());
    for (float value : values) {
        if (std::isfinite(value)) {
            filtered.push_back(value);
        }
    }
    if (filtered.empty()) {
        return 0.0f;
    }
    std::sort(filtered.begin(), filtered.end());
    const float t = Clamp(percentile / 100.0f, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(std::floor(t * static_cast<float>(filtered.size() - 1)));
    return filtered[index];
}

float ComputeAutoExposure(const RgbImageF& image, float percentile) {
    std::vector<float> luminance(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    for (size_t i = 0; i < luminance.size(); ++i) {
        const float r = image.pixels[i * 3U + 0U];
        const float g = image.pixels[i * 3U + 1U];
        const float b = image.pixels[i * 3U + 2U];
        luminance[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    const float p = ComputePercentile(luminance, percentile);
    return 1.0f / std::max(p, 1.0e-6f);
}

void SaveToneMappedRgbPng(const std::filesystem::path& path, const RgbImageF& image, float exposure) {
    EnsureDirectory(path.parent_path());
    std::vector<float> mapped(image.pixels.size(), 0.0f);
    constexpr float kInvGamma = 1.0f / 2.2f;
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        const float exposed = std::max(0.0f, image.pixels[i] * exposure);
        const float reinhard = exposed / (1.0f + exposed);
        mapped[i] = std::pow(reinhard, kInvGamma);
    }
    const std::vector<uint8_t> bytes = FloatRgbToU8(mapped);
    if (stbi_write_png(path.string().c_str(), image.width, image.height, 3, bytes.data(), image.width * 3) == 0) {
        throw std::runtime_error("Failed to write RGB PNG: " + path.string());
    }
}

std::vector<float> MakeSpectrumLogVisualization(
    const std::vector<std::complex<double>>& spectrum,
    int width,
    int height) {
    std::vector<float> magnitude(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        magnitude[i] = static_cast<float>(std::abs(spectrum[i]));
    }
    return MakeShiftedLogVisualization(magnitude, width, height);
}

std::vector<float> MakeSpectrumLogVisualization(
    const std::vector<float2>& spectrum,
    int width,
    int height) {
    std::vector<float> magnitude(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        magnitude[i] = std::sqrt(spectrum[i].x * spectrum[i].x + spectrum[i].y * spectrum[i].y);
    }
    return MakeShiftedLogVisualization(magnitude, width, height);
}

std::vector<float> MakeShiftedLogVisualization(
    const std::vector<float>& values,
    int width,
    int height) {
    return NormalizeLog(FftShift(values, width, height));
}

void WriteCaseMetricsJson(const CaseReport& report) {
    EnsureDirectory(report.case_dir);
    std::ofstream output(report.case_dir / "metrics.json");
    if (!output) {
        throw std::runtime_error("Failed to open metrics.json for writing");
    }
    output << "{\n"
           << "  \"case_id\": \"" << JsonEscape(CaseId(report)) << "\",\n"
           << "  \"width\": " << report.spec.width << ",\n"
           << "  \"height\": " << report.spec.height << ",\n"
           << "  \"image_type\": \"" << JsonEscape(report.spec.image_type) << "\",\n"
           << "  \"variant_id\": " << report.spec.variant_id << ",\n"
           << "  \"variant_name\": \"" << JsonEscape(report.variant_name) << "\",\n"
           << "  \"seed\": " << report.spec.seed << ",\n"
           << "  \"normalization\": \"inverse scales by 1/(width*height) only\",\n"
           << "  \"comparison_note\": \"LUT and approximate trig differ only in twiddle-factor implementation.\",\n"
           << "  \"mode_metrics\": {\n";
    for (size_t i = 0; i < report.mode_reports.size(); ++i) {
        const ModeReport& mode_report = report.mode_reports[i];
        output << "    \"" << JsonEscape(mode_report.mode) << "\": ";
        WriteJsonMetricSet(output, mode_report.metrics);
        if (i + 1 != report.mode_reports.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  },\n"
           << "  \"cross_metrics\": {\n"
           << "    \"mse_fast_vs_lut\": " << FormatDouble(report.cross_metrics.mse_fast_vs_lut) << ",\n"
           << "    \"mae_fast_vs_lut\": " << FormatDouble(report.cross_metrics.mae_fast_vs_lut) << ",\n"
           << "    \"max_abs_fast_vs_lut\": " << FormatDouble(report.cross_metrics.max_abs_fast_vs_lut) << ",\n"
           << "    \"relative_l2_fast_vs_lut\": " << FormatDouble(report.cross_metrics.relative_l2_fast_vs_lut) << "\n"
           << "  }\n"
           << "}\n";
}

void WriteSummaryCsv(const std::filesystem::path& path, const std::vector<CaseReport>& reports) {
    EnsureDirectory(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open summary CSV: " + path.string());
    }
    output << "width,height,image_type,variant_id,variant_name,seed,mode,"
           << "mse_vs_ref,rmse_vs_ref,mae_vs_ref,max_abs_error,psnr_vs_ref,relative_l2_error,"
           << "mean_abs_error_in_spectrum,max_abs_error_in_spectrum,"
           << "mse_fast_vs_lut,mae_fast_vs_lut,max_abs_fast_vs_lut,relative_l2_fast_vs_lut,"
           << "mse_vs_original,mae_vs_original,max_abs_error_vs_original,case_dir\n";

    for (const CaseReport& report : reports) {
        for (const ModeReport& mode_report : report.mode_reports) {
            output << report.spec.width << ","
                   << report.spec.height << ","
                   << report.spec.image_type << ","
                   << report.spec.variant_id << ","
                   << report.variant_name << ","
                   << report.spec.seed << ","
                   << mode_report.mode << ","
                   << FormatDouble(mode_report.metrics.mse_vs_ref) << ","
                   << FormatDouble(mode_report.metrics.rmse_vs_ref) << ","
                   << FormatDouble(mode_report.metrics.mae_vs_ref) << ","
                   << FormatDouble(mode_report.metrics.max_abs_error) << ",";
            if (mode_report.metrics.psnr_is_infinite) {
                output << ",";
            } else {
                output << FormatDouble(mode_report.metrics.psnr_vs_ref) << ",";
            }
            output << FormatDouble(mode_report.metrics.relative_l2_error) << ","
                   << FormatDouble(mode_report.metrics.mean_abs_error_in_spectrum) << ","
                   << FormatDouble(mode_report.metrics.max_abs_error_in_spectrum) << ","
                   << FormatDouble(report.cross_metrics.mse_fast_vs_lut) << ","
                   << FormatDouble(report.cross_metrics.mae_fast_vs_lut) << ","
                   << FormatDouble(report.cross_metrics.max_abs_fast_vs_lut) << ","
                   << FormatDouble(report.cross_metrics.relative_l2_fast_vs_lut) << ","
                   << FormatDouble(mode_report.metrics.mse_vs_original) << ","
                   << FormatDouble(mode_report.metrics.mae_vs_original) << ","
                   << FormatDouble(mode_report.metrics.max_abs_error_vs_original) << ","
                   << report.case_dir.generic_string() << "\n";
        }
    }
}

void WriteSummaryJson(const std::filesystem::path& path, const std::vector<CaseReport>& reports) {
    EnsureDirectory(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open summary JSON: " + path.string());
    }
    output << "{\n"
           << "  \"cases\": [\n";
    for (size_t case_idx = 0; case_idx < reports.size(); ++case_idx) {
        const CaseReport& report = reports[case_idx];
        output << "    {\n"
               << "      \"width\": " << report.spec.width << ",\n"
               << "      \"height\": " << report.spec.height << ",\n"
               << "      \"image_type\": \"" << JsonEscape(report.spec.image_type) << "\",\n"
               << "      \"variant_id\": " << report.spec.variant_id << ",\n"
               << "      \"variant_name\": \"" << JsonEscape(report.variant_name) << "\",\n"
               << "      \"seed\": " << report.spec.seed << ",\n"
               << "      \"case_dir\": \"" << JsonEscape(report.case_dir.generic_string()) << "\",\n"
               << "      \"mode_metrics\": {\n";
        for (size_t i = 0; i < report.mode_reports.size(); ++i) {
            output << "        \"" << JsonEscape(report.mode_reports[i].mode) << "\": ";
            WriteJsonMetricSet(output, report.mode_reports[i].metrics);
            if (i + 1 != report.mode_reports.size()) {
                output << ",";
            }
            output << "\n";
        }
        output << "      },\n"
               << "      \"cross_metrics\": {\n"
               << "        \"mse_fast_vs_lut\": " << FormatDouble(report.cross_metrics.mse_fast_vs_lut) << ",\n"
               << "        \"mae_fast_vs_lut\": " << FormatDouble(report.cross_metrics.mae_fast_vs_lut) << ",\n"
               << "        \"max_abs_fast_vs_lut\": " << FormatDouble(report.cross_metrics.max_abs_fast_vs_lut) << ",\n"
               << "        \"relative_l2_fast_vs_lut\": " << FormatDouble(report.cross_metrics.relative_l2_fast_vs_lut) << "\n"
               << "      }\n"
               << "    }";
        if (case_idx + 1 != reports.size()) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]\n"
           << "}\n";
}

}  // namespace fftlut
