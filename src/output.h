#pragma once

#include "common.h"

namespace fftlut {

void EnsureDirectory(const std::filesystem::path& path);

void SaveUnitFloatImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height);

void SaveLinearAutoImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height);

void SaveLogAutoImagePng(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height);

void SaveNpyFloat32(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    int width,
    int height);

void SaveNpyFloat32(
    const std::filesystem::path& path,
    const std::vector<float>& pixels,
    const std::vector<int>& shape);

void SaveHdrRgbNpy(const std::filesystem::path& path, const RgbImageF& image);
void SaveToneMappedRgbPng(const std::filesystem::path& path, const RgbImageF& image, float exposure);
float ComputePercentile(const std::vector<float>& values, float percentile);
float ComputeAutoExposure(const RgbImageF& image, float percentile);

std::vector<float> MakeSpectrumLogVisualization(
    const std::vector<std::complex<double>>& spectrum,
    int width,
    int height);

std::vector<float> MakeSpectrumLogVisualization(
    const std::vector<float2>& spectrum,
    int width,
    int height);

std::vector<float> MakeShiftedLogVisualization(
    const std::vector<float>& values,
    int width,
    int height);

void WriteCaseMetricsJson(const CaseReport& report);
void WriteSummaryCsv(const std::filesystem::path& path, const std::vector<CaseReport>& reports);
void WriteSummaryJson(const std::filesystem::path& path, const std::vector<CaseReport>& reports);

}  // namespace fftlut
