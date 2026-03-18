#pragma once

#include "common.h"
#include "gpu_fft.h"

namespace fftlut {

struct RgbKernelSpectra {
    std::array<std::vector<std::complex<double>>, 3> cpu_ref;
    std::array<std::vector<float2>, 3> gpu_lut;
    std::array<std::vector<float2>, 3> gpu_fast;
};

std::vector<float> ExtractChannelPlane(const RgbImageF& image, int channel);
RgbImageF ComposeRgbImage(const std::array<std::vector<float>, 3>& planes, int width, int height);
std::vector<float> IfftShiftPlane(const std::vector<float>& input, int width, int height);
RgbImageF IfftShiftRgbImage(const RgbImageF& image);
RgbKernelSpectra BuildKernelSpectra(const RgbImageF& shifted_kernel, GpuFftPlan& gpu_plan);
RgbImageF RunCpuBloom(
    const RgbImageF& source,
    const std::array<std::vector<std::complex<double>>, 3>& kernel_spectra);
RgbImageF RunGpuBloom(
    const RgbImageF& source,
    const std::array<std::vector<float2>, 3>& kernel_spectra,
    GpuFftPlan& gpu_plan,
    GpuTwiddleMode mode);
std::vector<float> ComputeLuminanceImage(const RgbImageF& image);
void ValidateFiniteImage(const RgbImageF& image, const std::string& label);
void RunBloomConvolutionSelfTest();

}  // namespace fftlut
