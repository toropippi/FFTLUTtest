#pragma once

#include "common.h"

namespace fftlut {

struct CpuFftResult {
    std::vector<std::complex<double>> spectrum;
    std::vector<double> reconstruction;
};

std::vector<std::complex<double>> Forward2DRealToComplex(
    const std::vector<float>& image,
    int width,
    int height);

std::vector<double> Inverse2DComplexToReal(
    const std::vector<std::complex<double>>& spectrum,
    int width,
    int height);

CpuFftResult RunCpuReference2D(const std::vector<float>& image, int width, int height);

}  // namespace fftlut
