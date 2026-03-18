#include "bloom.h"
#include "exr_io.h"
#include "hdr_scene.h"
#include "output.h"

#include <iostream>
#include <optional>

namespace fftlut {

struct BloomDebugConfig {
    std::filesystem::path kernel_path;
    std::filesystem::path output_dir;
    uint32_t seed = 1337;
    std::optional<float> display_exposure;
};

void PrintBloomDebugUsage() {
    std::cout
        << "Power-of-two square RGB HDR FFT bloom debug tool\n"
        << "Options:\n"
        << "  --kernel <path>           Required RGB kernel path (.bin or .exr)\n"
        << "  --kernel-bin <path>       Alias for --kernel with raw float32 RGB .bin\n"
        << "  --kernel-exr <path>       Alias for --kernel with RGB EXR\n"
        << "  --output-dir <path>       Required output directory\n"
        << "  --seed <uint>             Optional deterministic scene seed\n"
        << "  --display-exposure <f>    Optional shared exposure override\n"
        << "  --help\n";
}

BloomDebugConfig ParseBloomDebugArgs(int argc, char** argv) {
    BloomDebugConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            PrintBloomDebugUsage();
            std::exit(0);
        } else if (arg == "--kernel") {
            config.kernel_path = next_value("--kernel");
        } else if (arg == "--kernel-bin") {
            config.kernel_path = next_value("--kernel-bin");
        } else if (arg == "--kernel-exr") {
            config.kernel_path = next_value("--kernel-exr");
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

double MeanValue(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value);
    }
    return sum / static_cast<double>(values.size());
}

double MaxValue(const std::vector<float>& values) {
    double max_value = 0.0;
    for (float value : values) {
        max_value = std::max(max_value, static_cast<double>(value));
    }
    return max_value;
}

void WriteBloomDebugReportJson(
    const std::filesystem::path& path,
    const BloomDebugConfig& config,
    int width,
    int height,
    float display_exposure,
    float kernel_exposure,
    const std::vector<float>& diff_lut_vs_ref,
    const std::vector<float>& diff_fast_vs_ref,
    const std::vector<float>& diff_fast_vs_lut) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write bloom debug report: " + path.string());
    }

    output << "{\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"kernel_path\": \"" << JsonEscape(config.kernel_path.generic_string()) << "\",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"display_exposure\": " << FormatDouble(display_exposure) << ",\n"
           << "  \"kernel_display_exposure\": " << FormatDouble(kernel_exposure) << ",\n"
           << "  \"notes\": [\n"
           << "    \"Kernel is ifftshifted before FFT so the centered PSF becomes the FFT origin.\",\n"
           << "    \"GPU LUT and GPU fast differ only in twiddle-factor generation.\"\n"
           << "  ],\n"
           << "  \"luminance_absdiff\": {\n"
           << "    \"lut_vs_ref_mean\": " << FormatDouble(MeanValue(diff_lut_vs_ref)) << ",\n"
           << "    \"lut_vs_ref_max\": " << FormatDouble(MaxValue(diff_lut_vs_ref)) << ",\n"
           << "    \"fast_vs_ref_mean\": " << FormatDouble(MeanValue(diff_fast_vs_ref)) << ",\n"
           << "    \"fast_vs_ref_max\": " << FormatDouble(MaxValue(diff_fast_vs_ref)) << ",\n"
           << "    \"fast_vs_lut_mean\": " << FormatDouble(MeanValue(diff_fast_vs_lut)) << ",\n"
           << "    \"fast_vs_lut_max\": " << FormatDouble(MaxValue(diff_fast_vs_lut)) << "\n"
           << "  }\n"
           << "}\n";
}

int RunBloomDebugProgram(int argc, char** argv) {
    const BloomDebugConfig config = ParseBloomDebugArgs(argc, argv);
    EnsureDirectory(config.output_dir);

    RunBloomConvolutionSelfTest();

    const RgbImageF kernel = LoadRgbImageAuto(config.kernel_path);
    if (!IsPowerOfTwo(kernel.width) || !IsPowerOfTwo(kernel.height)) {
        throw std::runtime_error("Kernel image dimensions must be powers of two.");
    }
    if (kernel.width != kernel.height) {
        throw std::runtime_error("Kernel image must be square for the current bloom debug tool.");
    }

    const int width = kernel.width;
    const int height = kernel.height;

    const BloomDebugScene scene = GenerateBloomDebugScene(width, height, config.seed);
    const RgbImageF shifted_kernel = IfftShiftRgbImage(kernel);

    GpuFftPlan gpu_plan(width, height);
    const RgbKernelSpectra kernel_spectra = BuildKernelSpectra(shifted_kernel, gpu_plan);

    const RgbImageF cpu_ref = RunCpuBloom(scene.image, kernel_spectra.cpu_ref);
    const RgbImageF gpu_lut = RunGpuBloom(scene.image, kernel_spectra.gpu_lut, gpu_plan, GpuTwiddleMode::Lut);
    const RgbImageF gpu_fast = RunGpuBloom(scene.image, kernel_spectra.gpu_fast, gpu_plan, GpuTwiddleMode::Fast);

    ValidateFiniteImage(cpu_ref, "cpu_ref");
    ValidateFiniteImage(gpu_lut, "gpu_lut");
    ValidateFiniteImage(gpu_fast, "gpu_fast");

    const float display_exposure =
        config.display_exposure.has_value()
            ? *config.display_exposure
            : ComputeAutoExposure(cpu_ref, 99.5f);
    const float kernel_exposure = ComputeAutoExposure(kernel, 99.5f);

    const std::vector<float> lum_ref = ComputeLuminanceImage(cpu_ref);
    const std::vector<float> lum_lut = ComputeLuminanceImage(gpu_lut);
    const std::vector<float> lum_fast = ComputeLuminanceImage(gpu_fast);
    const std::vector<float> diff_lut_vs_ref = MakeAbsDiff(lum_lut, lum_ref);
    const std::vector<float> diff_fast_vs_ref = MakeAbsDiff(lum_fast, lum_ref);
    const std::vector<float> diff_fast_vs_lut = MakeAbsDiff(lum_fast, lum_lut);

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

    SaveLinearAutoImagePng(config.output_dir / "luminance_absdiff_lut_vs_ref.png", diff_lut_vs_ref, width, height);
    SaveLogAutoImagePng(config.output_dir / "luminance_absdiff_lut_vs_ref_log.png", diff_lut_vs_ref, width, height);
    SaveLinearAutoImagePng(config.output_dir / "luminance_absdiff_fast_vs_ref.png", diff_fast_vs_ref, width, height);
    SaveLogAutoImagePng(config.output_dir / "luminance_absdiff_fast_vs_ref_log.png", diff_fast_vs_ref, width, height);
    SaveLinearAutoImagePng(config.output_dir / "luminance_absdiff_fast_vs_lut.png", diff_fast_vs_lut, width, height);
    SaveLogAutoImagePng(config.output_dir / "luminance_absdiff_fast_vs_lut_log.png", diff_fast_vs_lut, width, height);

    WriteSpotMetadataJson(
        config.output_dir / "source_spots.json",
        scene.spots,
        width,
        height,
        config.seed);
    WriteBloomDebugReportJson(
        config.output_dir / "bloom_debug_report.json",
        config,
        width,
        height,
        display_exposure,
        kernel_exposure,
        diff_lut_vs_ref,
        diff_fast_vs_ref,
        diff_fast_vs_lut);

    std::cout << "Wrote bloom debug output to " << config.output_dir.string() << "\n";
    return 0;
}

}  // namespace fftlut

int main(int argc, char** argv) {
    try {
        return fftlut::RunBloomDebugProgram(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
