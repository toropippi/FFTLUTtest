#pragma once

#include "common.h"

namespace fftlut {

enum class GpuTwiddleMode {
    Lut,
    Fast,
};

struct GpuFftResult {
    std::vector<float2> spectrum;
    std::vector<float> reconstruction;
};

class GpuFftPlan {
public:
    GpuFftPlan(int width, int height);
    ~GpuFftPlan();

    GpuFftPlan(const GpuFftPlan&) = delete;
    GpuFftPlan& operator=(const GpuFftPlan&) = delete;

    std::vector<float2> Forward(const std::vector<float>& image, GpuTwiddleMode mode);
    std::vector<float> Inverse(const std::vector<float2>& spectrum, GpuTwiddleMode mode);
    std::vector<float> Convolve(
        const std::vector<float>& image,
        const std::vector<float2>& kernel_spectrum,
        GpuTwiddleMode mode);
    GpuFftResult Run(const std::vector<float>& image, GpuTwiddleMode mode);

private:
    void UploadRealInput(const std::vector<float>& image);
    void UploadSpectrumToPrimary(const std::vector<float2>& spectrum);
    std::vector<float2> DownloadPrimarySpectrum() const;
    std::vector<float> DownloadPrimaryReal() const;
    void RunForward(GpuTwiddleMode mode);
    void RunInverse(GpuTwiddleMode mode);
    void MultiplyPrimaryByHostSpectrum(const std::vector<float2>& other);
    void RunRowPass(
        const float2* src,
        float2* dst,
        int row_length,
        int row_count,
        bool inverse,
        GpuTwiddleMode mode,
        const float2* lut);
    void LaunchStages(
        float2* buffer,
        int row_length,
        int row_count,
        bool inverse,
        GpuTwiddleMode mode,
        const float2* lut);

    int width_ = 0;
    int height_ = 0;
    int log2_width_ = 0;
    int log2_height_ = 0;
    float* d_real_input_ = nullptr;
    float2* d_buffer_a_ = nullptr;
    float2* d_buffer_b_ = nullptr;
    float2* d_multiply_buffer_ = nullptr;
    float2* d_lut_width_ = nullptr;
    float2* d_lut_height_ = nullptr;
};

using GpuFftExperiment = GpuFftPlan;

}  // namespace fftlut
