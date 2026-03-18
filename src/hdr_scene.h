#pragma once

#include "common.h"

namespace fftlut {

struct BloomDebugScene {
    RgbImageF image;
    std::vector<HdrSpotMetadata> spots;
};

BloomDebugScene GenerateBloomDebugScene(int width, int height, uint32_t seed);
void WriteSpotMetadataJson(
    const std::filesystem::path& path,
    const std::vector<HdrSpotMetadata>& spots,
    int width,
    int height,
    uint32_t seed);

}  // namespace fftlut
