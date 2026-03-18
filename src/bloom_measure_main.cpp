#include "bloom.h"
#include "cpu_fft.h"
#include "exr_io.h"
#include "hdr_scene.h"
#include "output.h"

#include <iostream>
#include <optional>

namespace fftlut {

namespace {

struct BloomMeasureConfig {
    std::filesystem::path kernel_path;
    std::filesystem::path output_dir;
    uint32_t seed = 1337;
    std::optional<float> display_exposure;
};

struct BloomMetricSet {
    double mse_vs_ref = 0.0;
    double rmse_vs_ref = 0.0;
    double mae_vs_ref = 0.0;
    double max_abs_error = 0.0;
    double psnr_vs_ref = 0.0;
    bool psnr_is_infinite = false;
    double relative_l2_error = 0.0;
    double mean_abs_error_in_spectrum = 0.0;
    double max_abs_error_in_spectrum = 0.0;
    double mse_vs_source = 0.0;
    double mae_vs_source = 0.0;
    double max_abs_error_vs_source = 0.0;
};

struct BloomCrossMetricSet {
    double mse_fast_vs_lut = 0.0;
    double mae_fast_vs_lut = 0.0;
    double max_abs_fast_vs_lut = 0.0;
    double relative_l2_fast_vs_lut = 0.0;
};

struct ErrorAccumulator {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    double ref_sq = 0.0;
};

using CpuRgbSpectra = std::array<std::vector<std::complex<double>>, 3>;

void PrintUsage() {
    std::cout
        << "Production-style RGB HDR FFT bloom measurement runner\n"
        << "Options:\n"
        << "  --kernel <path>           Required RGB kernel path (.bin or .exr)\n"
        << "  --kernel-bin <path>       Alias for --kernel with raw float32 RGB .bin\n"
        << "  --kernel-exr <path>       Alias for --kernel with RGB EXR\n"
        << "  --output-dir <path>       Required output directory\n"
        << "  --seed <uint>             Optional deterministic scene seed\n"
        << "  --display-exposure <f>    Optional shared exposure override\n"
        << "  --help\n";
}

BloomMeasureConfig ParseArgs(int argc, char** argv) {
    BloomMeasureConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            PrintUsage();
            std::exit(0);
        } else if (arg == "--kernel" || arg == "--kernel-bin" || arg == "--kernel-exr") {
            config.kernel_path = next_value(arg.c_str());
        } else if (arg == "--output-dir") {
            config.output_dir = next_value("--output-dir");
        } else if (arg == "--seed") {
            config.seed = static_cast<uint32_t>(std::stoul(next_value("--seed")));
        } else if (arg == "--display-exposure") {
            config.display_exposure = std::stof(next_value("--display-exposure"));
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (config.kernel_path.empty()) {
        throw std::invalid_argument("--kernel is required");
    }
    if (config.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }

    config.kernel_path = std::filesystem::absolute(config.kernel_path);
    config.output_dir = std::filesystem::absolute(config.output_dir);
    return config;
}

ErrorAccumulator AccumulateErrors(const std::vector<float>& reference, const std::vector<float>& test) {
    if (reference.size() != test.size()) {
        throw std::runtime_error("Mismatched vector sizes for bloom metrics");
    }

    ErrorAccumulator acc;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref_value = static_cast<double>(reference[i]);
        const double test_value = static_cast<double>(test[i]);
        const double diff = test_value - ref_value;
        const double abs_diff = std::abs(diff);
        acc.sum_sq += diff * diff;
        acc.sum_abs += abs_diff;
        acc.max_abs = std::max(acc.max_abs, abs_diff);
        acc.ref_sq += ref_value * ref_value;
    }
    return acc;
}

CpuRgbSpectra BuildCpuRgbSpectra(const RgbImageF& image) {
    CpuRgbSpectra spectra;
    for (int channel = 0; channel < 3; ++channel) {
        spectra[static_cast<size_t>(channel)] =
            Forward2DRealToComplex(ExtractChannelPlane(image, channel), image.width, image.height);
    }
    return spectra;
}

std::pair<double, double> ComputeSpectrumErrors(
    const CpuRgbSpectra& reference,
    const CpuRgbSpectra& test) {
    double sum_abs = 0.0;
    double max_abs = 0.0;
    size_t count = 0;
    for (int channel = 0; channel < 3; ++channel) {
        const auto& ref_channel = reference[static_cast<size_t>(channel)];
        const auto& test_channel = test[static_cast<size_t>(channel)];
        if (ref_channel.size() != test_channel.size()) {
            throw std::runtime_error("Mismatched RGB spectrum sizes for bloom metrics");
        }
        for (size_t i = 0; i < ref_channel.size(); ++i) {
            const double abs_error = std::abs(test_channel[i] - ref_channel[i]);
            sum_abs += abs_error;
            max_abs = std::max(max_abs, abs_error);
        }
        count += ref_channel.size();
    }
    return {count == 0 ? 0.0 : (sum_abs / static_cast<double>(count)), max_abs};
}

BloomMetricSet FinalizeMetricSet(
    const ErrorAccumulator& ref_acc,
    const ErrorAccumulator& source_acc,
    size_t element_count,
    const std::pair<double, double>& spectrum_errors) {
    BloomMetricSet metrics;
    const double count = static_cast<double>(element_count);
    metrics.mse_vs_ref = ref_acc.sum_sq / count;
    metrics.rmse_vs_ref = std::sqrt(metrics.mse_vs_ref);
    metrics.mae_vs_ref = ref_acc.sum_abs / count;
    metrics.max_abs_error = ref_acc.max_abs;
    metrics.relative_l2_error = std::sqrt(ref_acc.sum_sq / std::max(ref_acc.ref_sq, 1.0e-30));
    metrics.mean_abs_error_in_spectrum = spectrum_errors.first;
    metrics.max_abs_error_in_spectrum = spectrum_errors.second;
    if (metrics.mse_vs_ref == 0.0) {
        metrics.psnr_is_infinite = true;
    } else {
        metrics.psnr_vs_ref = 10.0 * std::log10(1.0 / metrics.mse_vs_ref);
    }
    metrics.mse_vs_source = source_acc.sum_sq / count;
    metrics.mae_vs_source = source_acc.sum_abs / count;
    metrics.max_abs_error_vs_source = source_acc.max_abs;
    return metrics;
}

BloomMetricSet ComputeBloomMetricSet(
    const RgbImageF& reference,
    const RgbImageF& test,
    const CpuRgbSpectra& reference_spectrum,
    const CpuRgbSpectra& test_spectrum,
    const RgbImageF& source) {
    if (reference.width != test.width || reference.height != test.height ||
        source.width != test.width || source.height != test.height) {
        throw std::runtime_error("Bloom metric image sizes do not match");
    }

    const ErrorAccumulator ref_acc = AccumulateErrors(reference.pixels, test.pixels);
    const ErrorAccumulator source_acc = AccumulateErrors(source.pixels, test.pixels);
    return FinalizeMetricSet(
        ref_acc,
        source_acc,
        reference.pixels.size(),
        ComputeSpectrumErrors(reference_spectrum, test_spectrum));
}

BloomCrossMetricSet ComputeBloomCrossMetrics(const RgbImageF& gpu_fast, const RgbImageF& gpu_lut) {
    const ErrorAccumulator acc = AccumulateErrors(gpu_lut.pixels, gpu_fast.pixels);
    BloomCrossMetricSet metrics;
    const double count = static_cast<double>(gpu_fast.pixels.size());
    metrics.mse_fast_vs_lut = acc.sum_sq / count;
    metrics.mae_fast_vs_lut = acc.sum_abs / count;
    metrics.max_abs_fast_vs_lut = acc.max_abs;
    metrics.relative_l2_fast_vs_lut = std::sqrt(acc.sum_sq / std::max(acc.ref_sq, 1.0e-30));
    return metrics;
}

std::vector<float> MakeAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Diff array size mismatch");
    }
    std::vector<float> diff(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        diff[i] = std::abs(a[i] - b[i]);
    }
    return diff;
}

void SaveDiffArtifacts(
    const std::filesystem::path& output_dir,
    const std::string& base_name,
    const std::vector<float>& diff,
    int width,
    int height) {
    SaveNpyFloat32(output_dir / (base_name + ".npy"), diff, width, height);
    SaveLinearAutoImagePng(output_dir / (base_name + ".png"), diff, width, height);
    SaveLogAutoImagePng(output_dir / (base_name + "_log.png"), diff, width, height);
}

void WriteModeMetricsJson(std::ostream& output, const BloomMetricSet& metrics) {
    output << "{\n"
           << "      \"mse_vs_ref\": " << FormatDouble(metrics.mse_vs_ref) << ",\n"
           << "      \"rmse_vs_ref\": " << FormatDouble(metrics.rmse_vs_ref) << ",\n"
           << "      \"mae_vs_ref\": " << FormatDouble(metrics.mae_vs_ref) << ",\n"
           << "      \"max_abs_error\": " << FormatDouble(metrics.max_abs_error) << ",\n"
           << "      \"psnr_vs_ref\": ";
    if (metrics.psnr_is_infinite) {
        output << "null,\n";
    } else {
        output << FormatDouble(metrics.psnr_vs_ref) << ",\n";
    }
    output << "      \"psnr_vs_ref_is_infinite\": " << (metrics.psnr_is_infinite ? "true" : "false") << ",\n"
           << "      \"relative_l2_error\": " << FormatDouble(metrics.relative_l2_error) << ",\n"
           << "      \"mean_abs_error_in_spectrum\": " << FormatDouble(metrics.mean_abs_error_in_spectrum) << ",\n"
           << "      \"max_abs_error_in_spectrum\": " << FormatDouble(metrics.max_abs_error_in_spectrum) << ",\n"
           << "      \"mse_vs_source\": " << FormatDouble(metrics.mse_vs_source) << ",\n"
           << "      \"mae_vs_source\": " << FormatDouble(metrics.mae_vs_source) << ",\n"
           << "      \"max_abs_error_vs_source\": " << FormatDouble(metrics.max_abs_error_vs_source) << "\n"
           << "    }";
}

void WriteMetricsJson(
    const std::filesystem::path& path,
    const BloomMeasureConfig& config,
    int width,
    int height,
    float display_exposure,
    float kernel_exposure,
    const BloomMetricSet& cpu_metrics,
    const BloomMetricSet& lut_metrics,
    const BloomMetricSet& fast_metrics,
    const BloomCrossMetricSet& cross_metrics) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write bloom metrics JSON: " + path.string());
    }

    output << "{\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"kernel_path\": \"" << JsonEscape(config.kernel_path.generic_string()) << "\",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"display_exposure\": " << FormatDouble(display_exposure) << ",\n"
           << "  \"kernel_display_exposure\": " << FormatDouble(kernel_exposure) << ",\n"
           << "  \"mode_metrics\": {\n"
           << "    \"cpu_ref\": ";
    WriteModeMetricsJson(output, cpu_metrics);
    output << ",\n    \"gpu_lut\": ";
    WriteModeMetricsJson(output, lut_metrics);
    output << ",\n    \"gpu_fast\": ";
    WriteModeMetricsJson(output, fast_metrics);
    output << "\n  },\n"
           << "  \"cross_metrics\": {\n"
           << "    \"mse_fast_vs_lut\": " << FormatDouble(cross_metrics.mse_fast_vs_lut) << ",\n"
           << "    \"mae_fast_vs_lut\": " << FormatDouble(cross_metrics.mae_fast_vs_lut) << ",\n"
           << "    \"max_abs_fast_vs_lut\": " << FormatDouble(cross_metrics.max_abs_fast_vs_lut) << ",\n"
           << "    \"relative_l2_fast_vs_lut\": " << FormatDouble(cross_metrics.relative_l2_fast_vs_lut) << "\n"
           << "  },\n"
           << "  \"notes\": [\n"
           << "    \"Kernel is ifftshifted before FFT so the centered PSF becomes the FFT origin.\",\n"
           << "    \"GPU LUT and GPU fast differ only in twiddle-factor generation.\",\n"
           << "    \"Spectrum metrics are computed from CPU double FFTs of the final RGB bloom outputs.\"\n"
           << "  ]\n"
           << "}\n";
}

void WriteSummaryCsv(
    const std::filesystem::path& path,
    const BloomMeasureConfig& config,
    int width,
    int height,
    const BloomMetricSet& cpu_metrics,
    const BloomMetricSet& lut_metrics,
    const BloomMetricSet& fast_metrics,
    const BloomCrossMetricSet& cross_metrics) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write bloom summary CSV: " + path.string());
    }

    output << "width,height,seed,kernel_path,mode,"
           << "mse_vs_ref,rmse_vs_ref,mae_vs_ref,max_abs_error,psnr_vs_ref,relative_l2_error,"
           << "mean_abs_error_in_spectrum,max_abs_error_in_spectrum,"
           << "mse_fast_vs_lut,mae_fast_vs_lut,max_abs_fast_vs_lut,relative_l2_fast_vs_lut,"
           << "mse_vs_source,mae_vs_source,max_abs_error_vs_source\n";

    const std::array<std::pair<const char*, const BloomMetricSet*>, 3> rows = {{
        {"cpu_ref", &cpu_metrics},
        {"gpu_lut", &lut_metrics},
        {"gpu_fast", &fast_metrics},
    }};

    for (const auto& row : rows) {
        const BloomMetricSet& metrics = *row.second;
        output << width << ","
               << height << ","
               << config.seed << ","
               << "\"" << JsonEscape(config.kernel_path.generic_string()) << "\","
               << row.first << ","
               << FormatDouble(metrics.mse_vs_ref) << ","
               << FormatDouble(metrics.rmse_vs_ref) << ","
               << FormatDouble(metrics.mae_vs_ref) << ","
               << FormatDouble(metrics.max_abs_error) << ",";
        if (metrics.psnr_is_infinite) {
            output << ",";
        } else {
            output << FormatDouble(metrics.psnr_vs_ref) << ",";
        }
        output << FormatDouble(metrics.relative_l2_error) << ","
               << FormatDouble(metrics.mean_abs_error_in_spectrum) << ","
               << FormatDouble(metrics.max_abs_error_in_spectrum) << ","
               << FormatDouble(cross_metrics.mse_fast_vs_lut) << ","
               << FormatDouble(cross_metrics.mae_fast_vs_lut) << ","
               << FormatDouble(cross_metrics.max_abs_fast_vs_lut) << ","
               << FormatDouble(cross_metrics.relative_l2_fast_vs_lut) << ","
               << FormatDouble(metrics.mse_vs_source) << ","
               << FormatDouble(metrics.mae_vs_source) << ","
               << FormatDouble(metrics.max_abs_error_vs_source) << "\n";
    }
}

void WriteSummaryJson(
    const std::filesystem::path& path,
    const BloomMeasureConfig& config,
    int width,
    int height,
    const BloomMetricSet& cpu_metrics,
    const BloomMetricSet& lut_metrics,
    const BloomMetricSet& fast_metrics,
    const BloomCrossMetricSet& cross_metrics) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write bloom summary JSON: " + path.string());
    }

    output << "{\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"kernel_path\": \"" << JsonEscape(config.kernel_path.generic_string()) << "\",\n"
           << "  \"mode_metrics\": {\n"
           << "    \"cpu_ref\": ";
    WriteModeMetricsJson(output, cpu_metrics);
    output << ",\n    \"gpu_lut\": ";
    WriteModeMetricsJson(output, lut_metrics);
    output << ",\n    \"gpu_fast\": ";
    WriteModeMetricsJson(output, fast_metrics);
    output << "\n  },\n"
           << "  \"cross_metrics\": {\n"
           << "    \"mse_fast_vs_lut\": " << FormatDouble(cross_metrics.mse_fast_vs_lut) << ",\n"
           << "    \"mae_fast_vs_lut\": " << FormatDouble(cross_metrics.mae_fast_vs_lut) << ",\n"
           << "    \"max_abs_fast_vs_lut\": " << FormatDouble(cross_metrics.max_abs_fast_vs_lut) << ",\n"
           << "    \"relative_l2_fast_vs_lut\": " << FormatDouble(cross_metrics.relative_l2_fast_vs_lut) << "\n"
           << "  }\n"
           << "}\n";
}

int RunProgram(int argc, char** argv) {
    const BloomMeasureConfig config = ParseArgs(argc, argv);
    EnsureDirectory(config.output_dir);

    RunBloomConvolutionSelfTest();

    const RgbImageF kernel = LoadRgbImageAuto(config.kernel_path);
    if (!IsPowerOfTwo(kernel.width) || !IsPowerOfTwo(kernel.height) || kernel.width != kernel.height) {
        throw std::runtime_error("Kernel image must be square and power-of-two for bloom measurement.");
    }

    const BloomDebugScene scene = GenerateBloomDebugScene(kernel.width, kernel.height, config.seed);
    const RgbImageF shifted_kernel = IfftShiftRgbImage(kernel);

    GpuFftPlan gpu_plan(kernel.width, kernel.height);
    const RgbKernelSpectra kernel_spectra = BuildKernelSpectra(shifted_kernel, gpu_plan);

    const RgbImageF cpu_ref = RunCpuBloom(scene.image, kernel_spectra.cpu_ref);
    const RgbImageF gpu_lut = RunGpuBloom(scene.image, kernel_spectra.gpu_lut, gpu_plan, GpuTwiddleMode::Lut);
    const RgbImageF gpu_fast = RunGpuBloom(scene.image, kernel_spectra.gpu_fast, gpu_plan, GpuTwiddleMode::Fast);

    ValidateFiniteImage(cpu_ref, "cpu_ref");
    ValidateFiniteImage(gpu_lut, "gpu_lut");
    ValidateFiniteImage(gpu_fast, "gpu_fast");

    const CpuRgbSpectra cpu_ref_spectrum = BuildCpuRgbSpectra(cpu_ref);
    const CpuRgbSpectra gpu_lut_spectrum = BuildCpuRgbSpectra(gpu_lut);
    const CpuRgbSpectra gpu_fast_spectrum = BuildCpuRgbSpectra(gpu_fast);

    const BloomMetricSet cpu_metrics =
        ComputeBloomMetricSet(cpu_ref, cpu_ref, cpu_ref_spectrum, cpu_ref_spectrum, scene.image);
    const BloomMetricSet lut_metrics =
        ComputeBloomMetricSet(cpu_ref, gpu_lut, cpu_ref_spectrum, gpu_lut_spectrum, scene.image);
    const BloomMetricSet fast_metrics =
        ComputeBloomMetricSet(cpu_ref, gpu_fast, cpu_ref_spectrum, gpu_fast_spectrum, scene.image);
    const BloomCrossMetricSet cross_metrics = ComputeBloomCrossMetrics(gpu_fast, gpu_lut);

    const float display_exposure =
        config.display_exposure.has_value()
            ? *config.display_exposure
            : ComputeAutoExposure(cpu_ref, 99.5f);
    const float kernel_exposure = ComputeAutoExposure(kernel, 99.5f);

    const std::vector<float> lum_ref = ComputeLuminanceImage(cpu_ref);
    const std::vector<float> lum_lut = ComputeLuminanceImage(gpu_lut);
    const std::vector<float> lum_fast = ComputeLuminanceImage(gpu_fast);

    SaveHdrRgbNpy(config.output_dir / "source_hdr.npy", scene.image);
    SaveHdrRgbNpy(config.output_dir / "kernel_hdr.npy", kernel);
    SaveHdrRgbNpy(config.output_dir / "kernel_ifftshifted_hdr.npy", shifted_kernel);
    SaveHdrRgbNpy(config.output_dir / "bloom_cpu_ref_hdr.npy", cpu_ref);
    SaveHdrRgbNpy(config.output_dir / "bloom_gpu_lut_hdr.npy", gpu_lut);
    SaveHdrRgbNpy(config.output_dir / "bloom_gpu_fast_hdr.npy", gpu_fast);

    SaveToneMappedRgbPng(config.output_dir / "source_tonemapped.png", scene.image, display_exposure);
    SaveToneMappedRgbPng(config.output_dir / "kernel_tonemapped.png", kernel, kernel_exposure);
    SaveToneMappedRgbPng(config.output_dir / "bloom_cpu_ref_tonemapped.png", cpu_ref, display_exposure);
    SaveToneMappedRgbPng(config.output_dir / "bloom_gpu_lut_tonemapped.png", gpu_lut, display_exposure);
    SaveToneMappedRgbPng(config.output_dir / "bloom_gpu_fast_tonemapped.png", gpu_fast, display_exposure);

    SaveDiffArtifacts(
        config.output_dir,
        "luminance_absdiff_lut_vs_ref",
        MakeAbsDiff(lum_lut, lum_ref),
        kernel.width,
        kernel.height);
    SaveDiffArtifacts(
        config.output_dir,
        "luminance_absdiff_fast_vs_ref",
        MakeAbsDiff(lum_fast, lum_ref),
        kernel.width,
        kernel.height);
    SaveDiffArtifacts(
        config.output_dir,
        "luminance_absdiff_fast_vs_lut",
        MakeAbsDiff(lum_fast, lum_lut),
        kernel.width,
        kernel.height);

    WriteSpotMetadataJson(
        config.output_dir / "source_spots.json",
        scene.spots,
        kernel.width,
        kernel.height,
        config.seed);
    WriteMetricsJson(
        config.output_dir / "metrics.json",
        config,
        kernel.width,
        kernel.height,
        display_exposure,
        kernel_exposure,
        cpu_metrics,
        lut_metrics,
        fast_metrics,
        cross_metrics);
    WriteSummaryCsv(
        config.output_dir / "summary.csv",
        config,
        kernel.width,
        kernel.height,
        cpu_metrics,
        lut_metrics,
        fast_metrics,
        cross_metrics);
    WriteSummaryJson(
        config.output_dir / "summary.json",
        config,
        kernel.width,
        kernel.height,
        cpu_metrics,
        lut_metrics,
        fast_metrics,
        cross_metrics);

    std::cout << "Wrote bloom measurement output to " << config.output_dir.string() << "\n";
    return 0;
}

}  // namespace

}  // namespace fftlut

int main(int argc, char** argv) {
    try {
        return fftlut::RunProgram(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
