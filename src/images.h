#pragma once

#include "common.h"

namespace fftlut {

struct GeneratedImage {
    std::vector<float> pixels;
    std::string variant_name;
    std::string description;
};

std::vector<ImagePreset> EnumerateImagePresets();
std::vector<ImagePreset> EnumerateRepresentativeLargePresets();

GeneratedImage GenerateImage(
    const std::string& image_type,
    int variant_id,
    int width,
    int height,
    uint32_t seed);

}  // namespace fftlut
