#pragma once

#include "common.h"

namespace fftlut {

RgbImageF LoadRgbExr(const std::filesystem::path& path);
RgbImageF LoadRgbFloatBin(const std::filesystem::path& path);
RgbImageF LoadRgbImageAuto(const std::filesystem::path& path);

}  // namespace fftlut
