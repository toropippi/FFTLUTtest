#include "cpu_fft.h"
#include "gpu_fft.h"
#include "images.h"
#include "metrics.h"
#include "output.h"

#include <iostream>
#include <memory>

namespace fftlut {

namespace {

void PrintUsage() {
    std::cout
        << "CUDA FFT twiddle-factor comparison experiment\n"
        << "Options:\n"
        << "  --width <int>\n"
        << "  --height <int>\n"
        << "  --image-type <name>\n"
        << "  --variant <int>\n"
        << "  --seed <uint>\n"
        << "  --output-dir <path>\n"
        << "  --save-images <0|1>\n"
        << "  --save-spectrum <0|1>\n"
        << "  --run-all\n"
        << "  --help\n";
}

bool ParseBoolFlag(const std::string& value) {
    if (value == "1" || ToLower(value) == "true") {
        return true;
    }
    if (value == "0" || ToLower(value) == "false") {
        return false;
    }
    throw std::invalid_argument("Expected boolean value 0/1/true/false, got: " + value);
}

RunConfig ParseArgs(int argc, char** argv) {
    RunConfig config;
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
        } else if (arg == "--run-all") {
            config.run_all = true;
        } else if (arg == "--width") {
            config.width = std::stoi(next_value("--width"));
        } else if (arg == "--height") {
            config.height = std::stoi(next_value("--height"));
        } else if (arg == "--image-type") {
            config.image_type = next_value("--image-type");
        } else if (arg == "--variant") {
            config.variant_id = std::stoi(next_value("--variant"));
        } else if (arg == "--seed") {
            config.seed = static_cast<uint32_t>(std::stoul(next_value("--seed")));
        } else if (arg == "--output-dir") {
            config.output_dir = next_value("--output-dir");
        } else if (arg == "--save-images") {
            config.save_images = ParseBoolFlag(next_value("--save-images"));
        } else if (arg == "--save-spectrum") {
            config.save_spectrum = ParseBoolFlag(next_value("--save-spectrum"));
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    config.output_dir = std::filesystem::absolute(config.output_dir);
    return config;
}

std::vector<CaseSpec> BuildCases(const RunConfig& config) {
    std::vector<CaseSpec> cases;
    if (!config.run_all) {
        if (!IsPowerOfTwo(config.width) || !IsPowerOfTwo(config.height)) {
            throw std::invalid_argument("Width and height must both be powers of two.");
        }
        cases.push_back({
            config.width,
            config.height,
            config.image_type,
            config.variant_id,
            config.seed,
            config.save_images,
            config.save_spectrum,
        });
        return cases;
    }

    for (const ImagePreset& preset : EnumerateImagePresets()) {
        cases.push_back({
            256,
            256,
            preset.image_type,
            preset.variant_id,
            config.seed,
            config.save_images,
            config.save_spectrum,
        });
    }
    for (const ImagePreset& preset : EnumerateRepresentativeLargePresets()) {
        for (const int size : {512, 1024}) {
            cases.push_back({
                size,
                size,
                preset.image_type,
                preset.variant_id,
                config.seed,
                config.save_images,
                config.save_spectrum,
            });
        }
    }
    return cases;
}

std::vector<float> ToFloatVector(const std::vector<double>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]);
    }
    return output;
}

std::filesystem::path MakeCaseDir(const std::filesystem::path& root, const CaseSpec& spec) {
    std::ostringstream oss;
    oss << "case_" << spec.image_type << "_" << spec.variant_id << "_" << spec.width << "x" << spec.height;
    return root / oss.str();
}

void SaveDiffArtifacts(
    const std::filesystem::path& case_dir,
    const std::string& base_name,
    const std::vector<float>& diff,
    int width,
    int height) {
    SaveNpyFloat32(case_dir / (base_name + ".npy"), diff, width, height);
    SaveLinearAutoImagePng(case_dir / (base_name + ".png"), diff, width, height);
    SaveLogAutoImagePng(case_dir / (base_name + "_log.png"), diff, width, height);
}

CaseReport RunCase(
    const CaseSpec& spec,
    GpuFftExperiment& gpu_experiment,
    const std::filesystem::path& output_root) {
    GeneratedImage generated = GenerateImage(spec.image_type, spec.variant_id, spec.width, spec.height, spec.seed);
    const std::filesystem::path case_dir = MakeCaseDir(output_root, spec);
    EnsureDirectory(case_dir);

    std::cout << "Running " << spec.image_type << " variant " << spec.variant_id << " at "
              << spec.width << "x" << spec.height << "\n";

    if (spec.save_images) {
        SaveUnitFloatImagePng(case_dir / "original.png", generated.pixels, spec.width, spec.height);
    }

    const CpuFftResult cpu_result = RunCpuReference2D(generated.pixels, spec.width, spec.height);
    const GpuFftResult lut_result = gpu_experiment.Run(generated.pixels, GpuTwiddleMode::Lut);
    const GpuFftResult fast_result = gpu_experiment.Run(generated.pixels, GpuTwiddleMode::Fast);

    const MetricSet cpu_metrics = ComputeMetricSet(
        cpu_result.reconstruction,
        cpu_result.reconstruction,
        cpu_result.spectrum,
        cpu_result.spectrum,
        generated.pixels);
    const MetricSet lut_metrics = ComputeMetricSet(
        cpu_result.reconstruction,
        lut_result.reconstruction,
        cpu_result.spectrum,
        lut_result.spectrum,
        generated.pixels);
    const MetricSet fast_metrics = ComputeMetricSet(
        cpu_result.reconstruction,
        fast_result.reconstruction,
        cpu_result.spectrum,
        fast_result.spectrum,
        generated.pixels);
    const CrossMetricSet cross_metrics = ComputeCrossMetrics(fast_result.reconstruction, lut_result.reconstruction);

    const std::vector<float> cpu_recon_float = ToFloatVector(cpu_result.reconstruction);
    const std::vector<float> absdiff_lut_vs_ref =
        ComputeAbsDiffImage(cpu_result.reconstruction, lut_result.reconstruction);
    const std::vector<float> absdiff_fast_vs_ref =
        ComputeAbsDiffImage(cpu_result.reconstruction, fast_result.reconstruction);
    const std::vector<float> absdiff_fast_vs_lut =
        ComputeAbsDiffImage(fast_result.reconstruction, lut_result.reconstruction);

    if (spec.save_images) {
        SaveUnitFloatImagePng(case_dir / "recon_cpu_double.png", cpu_recon_float, spec.width, spec.height);
        SaveUnitFloatImagePng(case_dir / "recon_gpu_lut.png", lut_result.reconstruction, spec.width, spec.height);
        SaveUnitFloatImagePng(case_dir / "recon_gpu_fast.png", fast_result.reconstruction, spec.width, spec.height);

        SaveDiffArtifacts(case_dir, "absdiff_lut_vs_ref", absdiff_lut_vs_ref, spec.width, spec.height);
        SaveDiffArtifacts(case_dir, "absdiff_fast_vs_ref", absdiff_fast_vs_ref, spec.width, spec.height);
        SaveDiffArtifacts(case_dir, "absdiff_fast_vs_lut", absdiff_fast_vs_lut, spec.width, spec.height);
    }

    if (spec.save_spectrum) {
        const std::vector<float> spectrum_ref_log =
            MakeSpectrumLogVisualization(cpu_result.spectrum, spec.width, spec.height);
        const std::vector<float> spectrum_lut_log =
            MakeSpectrumLogVisualization(lut_result.spectrum, spec.width, spec.height);
        const std::vector<float> spectrum_fast_log =
            MakeSpectrumLogVisualization(fast_result.spectrum, spec.width, spec.height);
        const std::vector<float> spectrum_diff_fast_vs_ref =
            ComputeSpectrumAbsDiff(cpu_result.spectrum, fast_result.spectrum);
        const std::vector<float> spectrum_diff_lut_vs_ref =
            ComputeSpectrumAbsDiff(cpu_result.spectrum, lut_result.spectrum);

        SaveUnitFloatImagePng(case_dir / "spectrum_ref_log.png", spectrum_ref_log, spec.width, spec.height);
        SaveUnitFloatImagePng(case_dir / "spectrum_lut_log.png", spectrum_lut_log, spec.width, spec.height);
        SaveUnitFloatImagePng(case_dir / "spectrum_fast_log.png", spectrum_fast_log, spec.width, spec.height);
        SaveUnitFloatImagePng(
            case_dir / "spectrum_absdiff_fast_vs_ref_log.png",
            MakeShiftedLogVisualization(spectrum_diff_fast_vs_ref, spec.width, spec.height),
            spec.width,
            spec.height);

        SaveNpyFloat32(case_dir / "spectrum_absdiff_fast_vs_ref.npy", spectrum_diff_fast_vs_ref, spec.width, spec.height);
        SaveNpyFloat32(case_dir / "spectrum_absdiff_lut_vs_ref.npy", spectrum_diff_lut_vs_ref, spec.width, spec.height);
    }

    CaseReport report;
    report.spec = spec;
    report.variant_name = generated.variant_name;
    report.case_dir = case_dir;
    report.mode_reports = {
        {"cpu_ref", cpu_metrics},
        {"gpu_lut", lut_metrics},
        {"gpu_fast", fast_metrics},
    };
    report.cross_metrics = cross_metrics;
    WriteCaseMetricsJson(report);
    return report;
}

}  // namespace

}  // namespace fftlut

int main(int argc, char** argv) {
    try {
        const fftlut::RunConfig config = fftlut::ParseArgs(argc, argv);
        const std::vector<fftlut::CaseSpec> cases = fftlut::BuildCases(config);
        fftlut::EnsureDirectory(config.output_dir);

        std::map<std::pair<int, int>, std::unique_ptr<fftlut::GpuFftExperiment>> gpu_cache;
        std::vector<fftlut::CaseReport> reports;
        reports.reserve(cases.size());

        for (const fftlut::CaseSpec& spec : cases) {
            const auto key = std::make_pair(spec.width, spec.height);
            auto it = gpu_cache.find(key);
            if (it == gpu_cache.end()) {
                it = gpu_cache.emplace(key, std::make_unique<fftlut::GpuFftExperiment>(spec.width, spec.height)).first;
            }
            reports.push_back(fftlut::RunCase(spec, *it->second, config.output_dir));
        }

        fftlut::WriteSummaryCsv(config.output_dir / "summary.csv", reports);
        fftlut::WriteSummaryJson(config.output_dir / "summary.json", reports);
        std::cout << "Wrote output to " << config.output_dir.string() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
