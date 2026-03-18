#include "bloom.h"

#include "cpu_fft.h"

namespace fftlut {

namespace {

inline size_t PixelIndex(int x, int y, int width) {
    return (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3U;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Mismatched array sizes in MaxAbsDiff");
    }
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return max_abs;
}

}  // namespace

std::vector<float> ExtractChannelPlane(const RgbImageF& image, int channel) {
    if (channel < 0 || channel > 2) {
        throw std::invalid_argument("Invalid RGB channel index");
    }
    std::vector<float> plane(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    for (size_t i = 0; i < plane.size(); ++i) {
        plane[i] = image.pixels[i * 3U + static_cast<size_t>(channel)];
    }
    return plane;
}

RgbImageF ComposeRgbImage(const std::array<std::vector<float>, 3>& planes, int width, int height) {
    const size_t element_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (const auto& plane : planes) {
        if (plane.size() != element_count) {
            throw std::runtime_error("Plane size mismatch when composing RGB image");
        }
    }

    RgbImageF image;
    image.width = width;
    image.height = height;
    image.pixels.resize(element_count * 3U);
    for (size_t i = 0; i < element_count; ++i) {
        image.pixels[i * 3U + 0U] = planes[0][i];
        image.pixels[i * 3U + 1U] = planes[1][i];
        image.pixels[i * 3U + 2U] = planes[2][i];
    }
    return image;
}

std::vector<float> IfftShiftPlane(const std::vector<float>& input, int width, int height) {
    std::vector<float> output(input.size(), 0.0f);
    const int shift_x = width / 2;
    const int shift_y = height / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int src_x = (x + shift_x) % width;
            const int src_y = (y + shift_y) % height;
            output[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                input[static_cast<size_t>(src_y) * static_cast<size_t>(width) + static_cast<size_t>(src_x)];
        }
    }
    return output;
}

RgbImageF IfftShiftRgbImage(const RgbImageF& image) {
    std::array<std::vector<float>, 3> shifted;
    for (int channel = 0; channel < 3; ++channel) {
        shifted[static_cast<size_t>(channel)] = IfftShiftPlane(ExtractChannelPlane(image, channel), image.width, image.height);
    }
    return ComposeRgbImage(shifted, image.width, image.height);
}

RgbKernelSpectra BuildKernelSpectra(const RgbImageF& shifted_kernel, GpuFftPlan& gpu_plan) {
    RgbKernelSpectra spectra;
    for (int channel = 0; channel < 3; ++channel) {
        const std::vector<float> plane = ExtractChannelPlane(shifted_kernel, channel);
        spectra.cpu_ref[static_cast<size_t>(channel)] = Forward2DRealToComplex(plane, shifted_kernel.width, shifted_kernel.height);
        spectra.gpu_lut[static_cast<size_t>(channel)] = gpu_plan.Forward(plane, GpuTwiddleMode::Lut);
        spectra.gpu_fast[static_cast<size_t>(channel)] = gpu_plan.Forward(plane, GpuTwiddleMode::Fast);
    }
    return spectra;
}

RgbImageF RunCpuBloom(
    const RgbImageF& source,
    const std::array<std::vector<std::complex<double>>, 3>& kernel_spectra) {
    std::array<std::vector<float>, 3> output_planes;
    for (int channel = 0; channel < 3; ++channel) {
        const std::vector<float> plane = ExtractChannelPlane(source, channel);
        std::vector<std::complex<double>> spectrum = Forward2DRealToComplex(plane, source.width, source.height);
        for (size_t i = 0; i < spectrum.size(); ++i) {
            spectrum[i] *= kernel_spectra[static_cast<size_t>(channel)][i];
        }
        output_planes[static_cast<size_t>(channel)] = ToFloatVector(
            Inverse2DComplexToReal(spectrum, source.width, source.height));
    }
    return ComposeRgbImage(output_planes, source.width, source.height);
}

RgbImageF RunGpuBloom(
    const RgbImageF& source,
    const std::array<std::vector<float2>, 3>& kernel_spectra,
    GpuFftPlan& gpu_plan,
    GpuTwiddleMode mode) {
    std::array<std::vector<float>, 3> output_planes;
    for (int channel = 0; channel < 3; ++channel) {
        const std::vector<float> plane = ExtractChannelPlane(source, channel);
        output_planes[static_cast<size_t>(channel)] = gpu_plan.Convolve(
            plane,
            kernel_spectra[static_cast<size_t>(channel)],
            mode);
    }
    return ComposeRgbImage(output_planes, source.width, source.height);
}

std::vector<float> ComputeLuminanceImage(const RgbImageF& image) {
    std::vector<float> luminance(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    for (size_t i = 0; i < luminance.size(); ++i) {
        const float r = image.pixels[i * 3U + 0U];
        const float g = image.pixels[i * 3U + 1U];
        const float b = image.pixels[i * 3U + 2U];
        luminance[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    return luminance;
}

void ValidateFiniteImage(const RgbImageF& image, const std::string& label) {
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        if (!std::isfinite(image.pixels[i])) {
            throw std::runtime_error("Non-finite value found in " + label);
        }
    }
}

void RunBloomConvolutionSelfTest() {
    constexpr int kSize = 16;
    constexpr int kCenter = kSize / 2;

    std::vector<float> centered_kernel(static_cast<size_t>(kSize) * static_cast<size_t>(kSize), 0.0f);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const float dx = static_cast<float>(x - kCenter);
            const float dy = static_cast<float>(y - kCenter);
            centered_kernel[static_cast<size_t>(y) * static_cast<size_t>(kSize) + static_cast<size_t>(x)] =
                std::exp(-(dx * dx + dy * dy) / 6.0f);
        }
    }

    std::vector<float> delta(static_cast<size_t>(kSize) * static_cast<size_t>(kSize), 0.0f);
    delta[static_cast<size_t>(kCenter) * static_cast<size_t>(kSize) + static_cast<size_t>(kCenter)] = 1.0f;
    const std::vector<float> shifted_kernel = IfftShiftPlane(centered_kernel, kSize, kSize);

    std::vector<std::complex<double>> cpu_kernel_spectrum = Forward2DRealToComplex(shifted_kernel, kSize, kSize);
    std::vector<std::complex<double>> cpu_delta_spectrum = Forward2DRealToComplex(delta, kSize, kSize);
    for (size_t i = 0; i < cpu_delta_spectrum.size(); ++i) {
        cpu_delta_spectrum[i] *= cpu_kernel_spectrum[i];
    }
    const std::vector<double> cpu_out = Inverse2DComplexToReal(cpu_delta_spectrum, kSize, kSize);
    const std::vector<float> cpu_out_float = ToFloatVector(cpu_out);

    GpuFftPlan gpu_plan(kSize, kSize);
    const std::vector<float2> gpu_kernel_lut = gpu_plan.Forward(shifted_kernel, GpuTwiddleMode::Lut);
    const std::vector<float2> gpu_kernel_fast = gpu_plan.Forward(shifted_kernel, GpuTwiddleMode::Fast);
    const std::vector<float> gpu_out_lut = gpu_plan.Convolve(delta, gpu_kernel_lut, GpuTwiddleMode::Lut);
    const std::vector<float> gpu_out_fast = gpu_plan.Convolve(delta, gpu_kernel_fast, GpuTwiddleMode::Fast);

    const double cpu_error = MaxAbsDiff(cpu_out_float, centered_kernel);
    const double lut_error = MaxAbsDiff(gpu_out_lut, centered_kernel);
    const double fast_error = MaxAbsDiff(gpu_out_fast, centered_kernel);
    if (cpu_error > 1.0e-10 || lut_error > 1.0e-4 || fast_error > 2.5e-4) {
        std::ostringstream oss;
        oss << "Bloom convolution self-test failed. CPU=" << cpu_error
            << " LUT=" << lut_error
            << " FAST=" << fast_error;
        throw std::runtime_error(oss.str());
    }
}

}  // namespace fftlut
